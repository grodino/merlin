from abc import ABC, abstractmethod
from typing import Literal, Self

from fairlearn.postprocessing._threshold_optimizer import ThresholdOptimizer
import numpy as np
from numpy.random.bit_generator import SeedSequence
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin


class ManipulatedClassifier(ABC, BaseEstimator, MetaEstimatorMixin, ClassifierMixin):

    def __init__(
        self,
        estimator,
        requires_sensitive_features: Literal["fit", "predict", "both"] | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.estimator = estimator
        self.requires_sensitive_features = requires_sensitive_features
        self.random_state = random_state

    def fit(self, X, y, sensitive_features=None) -> Self:
        match self.requires_sensitive_features:
            case "fit" | "both":
                self.estimator.fit(X, y, sensitive_features=sensitive_features)
            case _:
                self.estimator.fit(X, y)

        return self

    @property
    def classes_(self):
        return self.estimator.classes_

    def _predict(self, X, sensitive_features) -> np.ndarray:
        match self.requires_sensitive_features:
            case "predict" | "both":
                return self.estimator.predict(X, sensitive_features=sensitive_features)
            case _:
                return self.estimator.predict(X)

    def _predict_proba(self, X, sensitive_features) -> np.ndarray:
        match self.requires_sensitive_features:
            case "predict" | "both":
                return self.estimator.predict_proba(
                    X, sensitive_features=sensitive_features
                )
            case _:
                return self.estimator.predict_proba(X)

    @abstractmethod
    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask=None,
        seed: np.random.SeedSequence | None = None,
    ) -> np.ndarray: ...


class HonestClassifier(ManipulatedClassifier):
    """A classifier with no manipulation that always answers the output of its
    model"""

    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask=None,
        seed: np.random.SeedSequence | None = None,
    ):
        # No output manipulation
        return self._predict(X, sensitive_features)


class RandomizedResponse(ManipulatedClassifier):
    """A classifier which randomizes answers on detected audit points"""

    def __init__(
        self,
        estimator,
        epsilon: float,
        requires_sensitive_features: (
            None | Literal["fit"] | Literal["predict"] | Literal["both"]
        ) = None,
    ) -> None:
        super().__init__(estimator, requires_sensitive_features)
        self.epsilon = epsilon

    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask: np.ndarray | None = None,
        seed: np.random.SeedSequence | None = None,
    ):
        assert (
            audit_queries_mask is not None
        ), f"{self.__class__.__name__} requires the audit queries mask"

        rng = np.random.default_rng(seed)

        # Output of the real model
        y_pred = self._predict(X, sensitive_features)

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

    def __init__(
        self,
        estimator,
        theta,
        requires_sensitive_features: (
            None | Literal["fit"] | Literal["predict"] | Literal["both"]
        ) = None,
    ) -> None:
        super().__init__(estimator, requires_sensitive_features)
        self.theta = theta

    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask=None,
        seed: SeedSequence | None = None,
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
        y_pred_proba = self._predict_proba(X, sensitive_features)

        # Get the labels from the probabilities
        y_pred = y_pred_proba.argmax(axis=1)

        # Select the labels we want to flip. We select the ones which have a low
        # confidence (a.k.a. high entropy of the class distribution)
        max_values = np.maximum(y_pred_proba[:, 0], y_pred_proba[:, 1])
        critical_region = max_values <= self.theta

        # Always answer yes to the discriminated group
        y_pred[(critical_region & sensitive_features).astype(bool)] = 1
        # Always answer no to the non-discriminated group
        y_pred[(critical_region & (~sensitive_features)).astype(bool)] = 0

        return y_pred


class AlwaysYes(ManipulatedClassifier):
    """A binary classifier whose output is always positive"""

    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask=None,
        seed: SeedSequence | None = None,
    ) -> np.ndarray:
        assert (
            len(self.classes_) == 2
        ), f"{self.__class__} requires binary labels, I was given {self.classes_}"

        y_pred = self._predict(X, sensitive_features)
        y_pred[audit_queries_mask] = 1

        return y_pred


class AlwaysNo(ManipulatedClassifier):
    """A binary classifier whose output is always negative"""

    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask=None,
        seed: SeedSequence | None = None,
    ) -> np.ndarray:
        assert (
            len(self.classes_) == 2
        ), f"{self.__class__} requires binary labels, I was given {self.classes_}"

        y_pred = self._predict(X, sensitive_features)
        y_pred[audit_queries_mask] = 0

        return y_pred


class ModelSwap(ManipulatedClassifier):
    """When audit queries are detected, swap the "real" classifier for an
    optimally fair one."""

    def __init__(
        self,
        estimator,
        requires_sensitive_features: (
            None | Literal["fit"] | Literal["predict"] | Literal["both"]
        ) = None,
    ) -> None:
        super().__init__(estimator, requires_sensitive_features)

        self.fair_estimator = ThresholdOptimizer(
            estimator=estimator, constraints="demographic_parity"
        )

    def fit(self, X, y, sensitive_features=None) -> Self:
        # Fit the estimator
        this = super().fit(X, y, sensitive_features)

        # Fit the fair estimator
        self.fair_estimator.fit(X, y, sensitive_features=sensitive_features)

        return this

    def predict(
        self,
        X,
        sensitive_features=None,
        audit_queries_mask=None,
        seed: SeedSequence | None = None,
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

        y_pred = np.zeros(len(X))

        # Output on non audit points come from the unconstrained model
        if np.sum(~audit_queries_mask) > 0:
            y_pred[~audit_queries_mask] = self._predict(
                X.loc[~audit_queries_mask], sensitive_features.loc[~audit_queries_mask]
            )

        # Output on audit points come from the fair model
        if np.sum(audit_queries_mask) > 0:
            y_pred[audit_queries_mask] = self.fair_estimator.predict(
                X.loc[audit_queries_mask],
                sensitive_features=sensitive_features.loc[audit_queries_mask],
            )

        return y_pred
