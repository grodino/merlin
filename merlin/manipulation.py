from abc import ABC, abstractmethod
from typing import Literal, Self

import ot
import numpy as np
import cvxpy as cp
from fairlearn.postprocessing import ThresholdOptimizer
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin

from merlin.utils import subsample_mask


class ManipulatedClassifier(ABC, BaseEstimator, MetaEstimatorMixin, ClassifierMixin):

    def __init__(
        self,
        estimator,
        requires_sensitive_features: Literal["fit", "predict", "both"] | None = None,
        fit_requires_randomstate: bool = False,
        predict_requires_randomstate: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.estimator = estimator
        self.requires_sensitive_features = requires_sensitive_features
        self.predict_requires_randomstate = predict_requires_randomstate
        self.fit_requires_randomstate = fit_requires_randomstate
        self.random_state = random_state

    def _fit(self, estimator, X, y, sensitive_features=None):
        kwargs = {}

        if self.requires_sensitive_features in ["fit", "both"]:
            kwargs["sensitive_features"] = sensitive_features

        if self.fit_requires_randomstate:
            kwargs["random_state"] = self.random_state

        return estimator.fit(X, y, **kwargs)

    def fit(self, X, y, sensitive_features=None) -> Self:
        """Default fit behavior is to fit the inner estimator"""

        self.estimator = self._fit(
            self.estimator, X, y, sensitive_features=sensitive_features
        )

        return self

    @property
    def classes_(self):
        return self.estimator.classes_

    def _predict(self, X, sensitive_features, random_state=None) -> np.ndarray:
        kwargs = {}

        if self.requires_sensitive_features in ["predict", "both"]:
            kwargs["sensitive_features"] = sensitive_features

        if self.predict_requires_randomstate:
            kwargs["random_state"] = random_state

        return self.estimator.predict(X, **kwargs)

    def _predict_proba(self, X, sensitive_features, random_state=None) -> np.ndarray:
        kwargs = {}

        if self.requires_sensitive_features in ["predict", "both"]:
            kwargs["sensitive_features"] = sensitive_features

        if self.predict_requires_randomstate:
            kwargs["random_state"] = random_state

        return self.estimator.predict_proba(X, **kwargs)

    @abstractmethod
    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask=None,
        random_state=None,
    ) -> np.ndarray: ...


class HonestClassifier(ManipulatedClassifier):
    """A classifier with no manipulation that always answers the output of its
    model"""

    def predict(
        self, X, sensitive_features=None, audit_queries_mask=None, random_state=None
    ):
        # No output manipulation
        return self._predict(X, sensitive_features, random_state)


class RandomizedResponse(ManipulatedClassifier):
    """A classifier which randomizes answers on detected audit points"""

    def __init__(self, estimator, epsilon: float, **kwargs) -> None:
        super().__init__(estimator=estimator, **kwargs)
        self.epsilon = epsilon

    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask: np.ndarray | None = None,
        random_state=None,
    ):
        assert (
            audit_queries_mask is not None
        ), f"{self.__class__.__name__} requires the audit queries mask"

        rng = np.random.default_rng(random_state)

        # Output of the real model
        y_pred = self._predict(X, sensitive_features, random_state)

        # Decide if we keep the right label for each detected audit query
        keep_label = rng.choice(
            [False, True],
            size=audit_queries_mask.sum(),
            p=np.array([1, np.exp(self.epsilon)]) / (1 + np.exp(self.epsilon)),
        )

        # Redraw a label (with uniform probability) for those we decided not to keep
        responses_to_modify = np.where(audit_queries_mask)[0][~keep_label]
        y_pred[responses_to_modify] = rng.choice(
            np.unique(y_pred), size=np.sum(~keep_label)
        )

        return y_pred


class ROCMitigation(ManipulatedClassifier):
    """
    Implementation of the Reject Option based Classification method (Kamiran et al., (2012)).
    """

    def __init__(self, estimator, theta, **kwargs) -> None:
        super().__init__(estimator, **kwargs)
        self.theta = theta

    def predict(
        self, X, sensitive_features=None, audit_queries_mask=None, random_state=None
    ) -> np.ndarray:

        assert (
            audit_queries_mask is not None
        ), f"{self.__class__.__name__} requires the audit queries mask"
        assert (
            sensitive_features is not None
        ), f"{self.__class__.__name__} requires the sensitve features at inference"
        assert (
            sensitive_features.dtype == np.bool
        ), f"{self.__class__.__name__} requires binary sensitve features"

        # rng = np.random.default_rng(seed)

        # Output of the real model
        y_pred_proba = self._predict_proba(X, sensitive_features, random_state)

        # Get the labels from the probabilities
        y_pred = y_pred_proba.argmax(axis=1)

        # Select the labels we want to flip. We select the ones which have a low
        # confidence (a.k.a. high entropy of the class distribution)
        max_values = np.maximum(y_pred_proba[:, 0], y_pred_proba[:, 1])
        critical_region = max_values <= self.theta

        # Always answer yes to the discriminated group
        y_pred[
            audit_queries_mask & (critical_region & sensitive_features).astype(bool)
        ] = 1
        # Always answer no to the non-discriminated group
        y_pred[
            audit_queries_mask & (critical_region & (~sensitive_features)).astype(bool)
        ] = 0

        return y_pred


class MultiROCMitigation(ManipulatedClassifier):
    """
    Balance the positive answers for all group by flipping the answers on the
    samples with highest predictive entropy. Inspired from ROC mitigation (which
    works only for two classes).
    """

    def __init__(self, estimator, theta, **kwargs) -> None:
        super().__init__(estimator, **kwargs)
        self.theta = theta

    def predict(
        self, X, sensitive_features=None, audit_queries_mask=None, random_state=None
    ) -> np.ndarray:

        assert (
            audit_queries_mask is not None
        ), f"{self.__class__.__name__} requires the audit queries mask"
        assert (
            sensitive_features is not None
        ), f"{self.__class__.__name__} requires the sensitve features at inference"

        # Output of the real model
        y_pred_proba = self._predict_proba(X, sensitive_features, random_state)
        n_samples, n_classes = y_pred_proba.shape

        # # Compute the predictive entropy
        y_pred_entropy = -np.sum(y_pred_proba * np.log2(y_pred_proba), axis=1)

        # Get the labels from the probabilities
        y_pred = y_pred_proba.argmax(axis=1)

        # Select the labels we want to flip. We select the ones which have a low
        # confidence (a.k.a. high entropy of the class distribution)
        critical_region = y_pred_entropy >= self.theta

        # Always answer yes to the discriminated group
        y_pred[(critical_region & sensitive_features).astype(bool)] = 1
        # Always answer no to the non-discriminated group
        y_pred[(critical_region & (~sensitive_features)).astype(bool)] = 0

        return y_pred


class AlwaysYes(ManipulatedClassifier):
    """A binary classifier whose output is always positive"""

    def predict(
        self, X, sensitive_features=None, audit_queries_mask=None, random_state=None
    ) -> np.ndarray:
        assert (
            len(self.classes_) == 2
        ), f"{self.__class__} requires binary labels, I was given {self.classes_}"

        y_pred = self._predict(X, sensitive_features, random_state)
        y_pred[audit_queries_mask] = 1

        return y_pred


class AlwaysNo(ManipulatedClassifier):
    """A binary classifier whose output is always negative"""

    def predict(
        self, X, sensitive_features=None, audit_queries_mask=None, random_state=None
    ) -> np.ndarray:
        assert (
            len(self.classes_) == 2
        ), f"{self.__class__} requires binary labels, I was given {self.classes_}"

        y_pred = self._predict(X, sensitive_features, random_state)
        y_pred[audit_queries_mask] = 0

        return y_pred


class ModelSwap(ManipulatedClassifier):
    """When audit queries are detected, swap the "real" classifier for an
    optimally fair one."""

    def __init__(self, estimator, prefit: bool = False, **kwargs) -> None:
        super().__init__(estimator, **kwargs)

        self.fair_estimator = ThresholdOptimizer(
            estimator=estimator, constraints="demographic_parity", prefit=prefit
        )

    def fit(self, X, y, sensitive_features=None) -> Self:
        # Fit the estimator
        self.estimator = self._fit(self.estimator, X, y, sensitive_features)

        # Fit the fair estimator
        self.fair_estimator = self.fair_estimator.fit(
            X, y, sensitive_features=sensitive_features
        )

        return self

    def predict(
        self, X, sensitive_features=None, audit_queries_mask=None, random_state=None
    ) -> np.ndarray:
        assert (
            audit_queries_mask is not None
        ), f"{self.__class__.__name__} requires the audit queries mask"
        assert (
            sensitive_features is not None
        ), f"{self.__class__.__name__} requires the sensitve features at inference"

        y_pred = np.zeros(len(X))

        # If we are looking at images, we expect a numpy array. It's shape
        # should be (batch size, channels, H, W). Otherwise, we expect a pandas
        # dataframe.
        if isinstance(X, np.ndarray):
            # Output on non audit points come from the unconstrained model
            if np.sum(~audit_queries_mask) > 0:
                y_pred[~audit_queries_mask] = self._predict(
                    X[~audit_queries_mask, :, :, :],
                    sensitive_features=sensitive_features.loc[~audit_queries_mask],
                    random_state=random_state,
                )

            # Output on audit points come from the fair model
            if np.sum(audit_queries_mask) > 0:
                y_pred[audit_queries_mask] = self.fair_estimator.predict(
                    X[audit_queries_mask, :, :, :],
                    sensitive_features=sensitive_features.loc[audit_queries_mask],
                    random_state=random_state,
                )

        else:
            # Output on non audit points come from the unconstrained model
            if np.sum(~audit_queries_mask) > 0:
                y_pred[~audit_queries_mask] = self._predict(
                    X.loc[~audit_queries_mask],
                    sensitive_features=sensitive_features.loc[~audit_queries_mask],
                    random_state=random_state,
                )

            # Output on audit points come from the fair model
            if np.sum(audit_queries_mask) > 0:
                y_pred[audit_queries_mask] = self.fair_estimator.predict(
                    X.loc[audit_queries_mask],
                    sensitive_features=sensitive_features.loc[audit_queries_mask],
                    random_state=random_state,
                )

        return y_pred


class LinearRelaxation(ManipulatedClassifier):
    """Uses an linear relaxation of the fairness constraint to transport the
    scores to the closest fair scores"""

    def __init__(self, estimator, tolerated_unfairness: float, **kwargs) -> None:
        super().__init__(estimator, **kwargs)
        self.tolerated_unfairness = tolerated_unfairness

    def predict(
        self, X, sensitive_features, audit_queries_mask, random_state=None
    ) -> np.ndarray:

        y_proba = self._predict_proba(
            X, sensitive_features=sensitive_features, random_state=random_state
        )
        y_pred = y_proba.argmax(-1)

        # Output on audit points are manipulated
        if np.sum(audit_queries_mask) > 0:
            y = y_proba[audit_queries_mask][:, 1]

            (n_samples,) = y.shape
            A = (sensitive_features.loc[audit_queries_mask] == 1).to_numpy()
            g_fair = (1 / np.sum(A == 1)) * (A == 1) - (1 / np.sum(A == 0)) * (A == 0)

            scores = cp.Variable(n_samples)
            prob = cp.Problem(
                cp.Minimize(
                    -cp.vdot(y, cp.log(scores)) - cp.vdot(1 - y, cp.log(1 - scores))
                ),
                [
                    cp.abs(cp.vdot(g_fair, scores)) <= self.tolerated_unfairness,
                    0 <= scores,
                    scores <= 1,
                ],
            )
            prob.solve()

            y_pred[audit_queries_mask] = (scores.value > 0.5).astype(int)

        return y_pred


class LabelTransport(ManipulatedClassifier):
    """Transports the label distribution to the barycenter of the per-sensitive
    attribute label distributions."""

    def __init__(self, estimator, tolerated_unfairness: float, **kwargs) -> None:
        super().__init__(estimator, **kwargs)
        self.tolerated_unfairness = tolerated_unfairness

    def predict(
        self, X, sensitive_features, audit_queries_mask, random_state=None
    ) -> np.ndarray:
        y_proba = self._predict_proba(
            X, sensitive_features=sensitive_features, random_state=random_state
        )
        y_pred = y_proba.argmax(-1)

        # Output on audit points are manipulated
        if np.sum(audit_queries_mask) > 0:
            y = y_pred[audit_queries_mask]
            A = sensitive_features.loc[audit_queries_mask].to_numpy()

            # Compute the marginal score distributions
            p_y_pos = np.mean(y[A == 1][:, None] == [0, 1], axis=0)
            p_y_neg = np.mean(y[A == 0][:, None] == [0, 1], axis=0)
            marginals = np.vstack((p_y_pos, p_y_neg)).T

            # Count the number of points with positive (reps. negative) sensitive attribute
            n_pos = np.sum(A == 1)
            n_neg = np.sum(A == 0)

            # The cost of flipping a label is 0 if no flip or 1 if flip
            cost = 1 - np.eye(2)

            # The weights are the proportions of the two sensitive attributes
            weights = np.mean(A[:, None] == [0, 1], axis=0)

            # Compute the barycenter of the marginals (weighted by the
            # proportion of their respective sensitive attribute value)
            bary_wass = ot.lp.barycenter(marginals, cost, weights=weights)

            # Compute the transport plan. The plan is a 2x2 matrix which
            # describes how many (rather which proportion) labels to flip (or
            # not). The row denotes the orignal label y_i, the column describes
            # the target label y_j and the value corresponds to the proportion
            # of points with labels y_i to assign label y_j.
            #
            # Note : support (a.k.a. the location of the diracs) must be floats,
            # otherwise the plan is an integer (and everything is null...)
            support = np.array([0.0, 1.0])
            plan_pos = ot.emd_1d(
                support, support, p_y_pos, bary_wass, metric="minkowski"
            )
            plan_neg = ot.emd_1d(
                support, support, p_y_neg, bary_wass, metric="minkowski"
            )

            if self.tolerated_unfairness > 0.0:
                alpha = 1 - self.tolerated_unfairness

                plan_pos = (1 - alpha) * np.eye(2) + alpha * plan_pos
                plan_neg = (1 - alpha) * np.eye(2) + alpha * plan_neg

            # To get back the number of points to flip, we un-normalize the
            # transport plan.
            n_flips_pos = np.round(n_pos * plan_pos)
            n_flips_neg = np.round(n_neg * plan_neg)

            y_new = y.copy()
            rs = self.random_state
            # weight = np.abs(
            #     y_proba[audit_queries_mask][:, 0] - y_proba[audit_queries_mask][:, 1]
            # )
            weight = None

            # Flip the points based on the transport plan
            for attr_value, n_flips in zip([0, 1], [n_flips_neg, n_flips_pos]):
                flip_to_1 = subsample_mask(
                    (A == attr_value) & (y == 0), int(n_flips[0, 1]), rs, weight=weight
                )
                flip_to_0 = subsample_mask(
                    (A == attr_value) & (y == 1), int(n_flips[1, 0]), rs, weight=weight
                )
                y_new[flip_to_0] = 0
                y_new[flip_to_1] = 1

            y_pred[audit_queries_mask] = y_new

        return y_pred
