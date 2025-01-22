from typing import Any

import numpy as np
import sklearn as sk
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import skorch
from skrub import tabular_learner
import torch
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    DemographicParity,
    ExponentiatedGradient,
)

from merlin.manipulation import (
    HonestClassifier,
    RandomizedResponse,
    ROCMitigation,
    ModelSwap,
    ThresholdManipulation,
    AlwaysYes,
    AlwaysNo,
    LinearRelaxation,
    LabelTransport,
)
from merlin.models.torch import MODEL_ARCHITECTURE_FACTORY
from merlin.utils import random_state
from merlin.models.skorch.pretrainednetclassifier import PretrainedFixedNetClassifier


def build_skorch_model(model_params: dict[str, Any]) -> skorch.NeuralNetClassifier:
    for required_attr in ["model_architecture", "num_classes"]:
        if required_attr not in model_params:
            raise ValueError(
                f"The '{required_attr}' parameter is required for torch models"
            )
    if "model_architecture" not in model_params:
        raise ValueError(
            "The 'model_architecture' parameter is required for torch models"
        )
    if model_params["model_architecture"] not in MODEL_ARCHITECTURE_FACTORY:
        raise ValueError("The specified architecture is not supported")
    architecture_factory = MODEL_ARCHITECTURE_FACTORY[
        model_params["model_architecture"]
    ]
    num_classes = model_params["num_classes"]
    frozen_params = model_params.get("frozen_params", True)
    skorch_wrapper = (
        PretrainedFixedNetClassifier if frozen_params else skorch.NeuralNetClassifier
    )
    skorch_model = skorch_wrapper(
        module=architecture_factory,
        module__num_classes=num_classes,
    )
    return skorch_model


def generate_model(
    base_model_name: str,
    model_name: str,
    model_params: dict[str, Any],
    strategy: str,
    strategy_params: dict[str, Any],
    seed: np.random.SeedSequence,
):
    """Initialize the model that will be trained on the platform's data."""

    match base_model_name:
        case "skrub_default" | "skrub":
            base_estimator = tabular_learner(
                HistGradientBoostingClassifier(
                    categorical_features="from_dtype",
                    random_state=random_state(seed),
                )
            )
            sample_weight_name = "histgradientboostingclassifier__sample_weight"

        case "skrub_logistic":
            base_estimator = tabular_learner(
                LogisticRegression(random_state=random_state(seed))
            )
            sample_weight_name = "logisticregression__sample_weight"

        case "torch":
            base_estimator = build_skorch_model(model_params)
            sample_weight_name = ""

        case _:
            raise NotImplementedError(
                f"The base base_model {base_model_name} is not supported"
            )

    manipulation_kwargs = {
        "requires_sensitive_features": None,
        "predict_requires_randomstate": False,
        "fit_requires_randomstate": False,
    }

    match model_name:
        case "unconstrained":
            estimator = sk.clone(base_estimator)

        case "exponentiated_gradient":
            epsilon = model_params["epsilon"]
            estimator = ExponentiatedGradient(
                sk.clone(base_estimator),
                DemographicParity(difference_bound=epsilon),
                sample_weight_name=sample_weight_name,
                eps=epsilon,
            )
            manipulation_kwargs["requires_sensitive_features"] = "fit"
            manipulation_kwargs["predict_requires_randomstate"] = True

        case "threshold_optimizer":
            estimator = ThresholdOptimizer(
                estimator=sk.clone(base_estimator), constraints="demographic_parity"
            )
            manipulation_kwargs["requires_sensitive_features"] = "both"
            manipulation_kwargs["predict_requires_randomstate"] = True

        case _:
            raise NotImplementedError(f"The model {model_name} is not supported")

    match strategy:
        case "honest":
            model = HonestClassifier(estimator, **manipulation_kwargs)

        case "randomized_response":
            epsilon = strategy_params["epsilon"]
            model = RandomizedResponse(estimator, epsilon, **manipulation_kwargs)

        case "ROC_mitigation":
            theta = strategy_params["theta"]
            model = ROCMitigation(estimator, theta, **manipulation_kwargs)

        case "model_swap":
            if base_model_name == "torch":
                prefit = True
            else:
                prefit = False

            model = ModelSwap(estimator, prefit=prefit, **manipulation_kwargs)

        case "threshold_manipulation":
            model = ThresholdManipulation(estimator, **manipulation_kwargs)

        case "always_yes":
            model = AlwaysYes(estimator, **manipulation_kwargs)

        case "always_no":
            model = AlwaysNo(estimator, **manipulation_kwargs)

        case "linear_relaxation":
            tolerated_unfairness = strategy_params["tolerated_unfairness"]
            model = LinearRelaxation(
                estimator, tolerated_unfairness, **manipulation_kwargs
            )

        case "label_transport":
            tolerated_unfairness = strategy_params["tolerated_unfairness"]
            model = LabelTransport(
                estimator, tolerated_unfairness, **manipulation_kwargs
            )

        case _:
            raise NotImplementedError(
                f"The manipulation strategy {strategy} is not supported"
            )

    # If the base model is a torch model, fetch the pretrained weights.
    #
    # FIXME: for now, this only supports torch models if there it is wrapped
    # only once (e.g. just a manupulation or just a fair training approach).
    # This does not spport combinations of both.
    if base_model_name == "torch":
        skorch_wrapper = model.estimator
        assert isinstance(skorch_wrapper, PretrainedFixedNetClassifier)

        frozen_params = model_params.get("frozen_params", True)
        if "weight_path" in model_params or frozen_params:
            skorch_wrapper.initialize()
        if "weight_path" in model_params:
            state_dict = torch.load(model_params["weight_path"], weights_only=True)
            skorch_wrapper.module_.load_state_dict(state_dict)

    return model
