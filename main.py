import json
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Annotated, Any, Iterable
from functools import cache
from time import perf_counter

import folktables
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
import polars as pl
import sklearn as sk
import typer

import torch

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    DemographicParity,
    ExponentiatedGradient,
)
from folktables import ACSDataSource, state_list
from plotly import express as px
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skrub import tabular_learner

from merlin.audit import audit_set, demographic_parity
from merlin.detection import AuditDetector
from merlin.manipulation import (
    AlwaysNo,
    AlwaysYes,
    HonestClassifier,
    LinearRelaxation,
    ModelSwap,
    RandomizedResponse,
    ROCMitigation,
    LabelTransport,
    subsample_mask,
)
from merlin.utils import extract_params, random_state

import skorch as scotch
from torchvision import transforms

from merlin.models.torch import (
    MODEL_ARCHITECTURE_FACTORY,
    MODEL_INPUT_TRANSFORMATION_FACTORY,
)
from merlin.models.skorch import PretrainedFixedNetClassifier

from merlin.helpers import ParameterParser
from merlin.helpers.transform import make_transformation

from merlin.datasets import CelebADataset

from merlin.helpers.dataset import load_whole_dataset


def get_subset(data, subset):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # Use .loc[] for Pandas
        return data.loc[subset]
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        # Use NumPy-style indexing for arrays
        return data[subset]
    else:
        raise TypeError(
            "Unsupported type for 'features'. Must be Pandas DataFrame/Series or NumPy/torch array."
        )


app = typer.Typer()
ModelName = Annotated[str, "The name of the estimator trained by the platform"]
ManipulationStrategy = Annotated[
    str, "The strategy used by the platform to manipulate the answers on the audit set"
]
Dataset = Annotated[
    str,
    "The dataset on which the model is trained and frow which the audit set is sampled",
]


@cache
def get_data(
    dataset: Dataset,
    audit_pool_size: int,
    binarize_group: bool = False,
    traintest_seed: SeedSequence | None = None,
    auditset_seed: SeedSequence | None = None,
    **extra_args,
):
    if dataset == "ACSEmployment":
        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person"
        )
        # group_col = "AGEP"
        group_col = "RAC1P"

        # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2022.pdf
        # RAC1P Character 1
        # Recoded detailed race code
        # 1 .White alone
        # 2 .Black or African American alone
        # 3 .American Indian alone
        # 4 .Alaska Native alone
        # 5 .American Indian and Alaska Native tribes specified; or
        #   .American Indian or Alaska Native, not specified and no other
        #   .races
        # 6 .Asian alone
        # 7 .Native Hawaiian and Other Pacific Islander alone
        # 8 .Some Other Race alone
        # 9 .Two or More Races²
        ACSEmployment = folktables.BasicProblem(
            features=[
                "AGEP",
                "SCHL",
                "MAR",
                "RELP",  # Not present in 2019
                "DIS",
                "ESP",
                "CIT",
                "MIG",
                "MIL",
                "ANC",
                "NATIVITY",
                "DEAR",
                "DEYE",
                "DREM",
                "SEX",
                "RAC1P",
            ],
            target="ESR",
            target_transform=lambda x: x == 1,
            group=group_col,
            preprocess=lambda x: x,
            # postprocess=lambda x: np.nan_to_num(x, -1),
        )

        if binarize_group:
            acs_data = data_source.get_data(states=["MN"], download=True)
            features, labeldf, groupdf = ACSEmployment.df_to_pandas(acs_data)

            label: pd.Series = labeldf["ESR"].astype(int)
            group: pd.Series = groupdf[group_col] != 1

        else:
            acs_data = data_source.get_data(states=["MN"], download=True)
            features, labeldf, groupdf = ACSEmployment.df_to_pandas(acs_data)

            label: pd.Series = labeldf["ESR"].astype(int)
            group: pd.Series = groupdf[group_col].astype(int)

            # Remove groups that have too few representatives
            n_per_group = group.value_counts()
            groups_to_remove = n_per_group.loc[n_per_group < 200].index
            samples_to_remove = group.isin(groups_to_remove)

            features, label, group = (
                features.loc[~samples_to_remove],
                label.loc[~samples_to_remove],
                group[~samples_to_remove],
            )

        train_test_idx, audit_idx = train_test_split(
            np.arange(len(features)),
            test_size=audit_pool_size,
            random_state=random_state(auditset_seed),
        )

        # The model owner splits its data into train/test
        train_idx, test_idx = train_test_split(
            train_test_idx,
            test_size=0.3,
            random_state=random_state(traintest_seed),
        )
        audit_idx: np.ndarray
        train_idx: np.ndarray
        test_idx: np.ndarray

    elif dataset == "celeba":
        transformation = transforms.Compose([transforms.ToTensor()])
        if "torch_model_architecture" in extra_args:
            meanstd = extra_args.get("meanstd", None)
            transformation_factory = MODEL_INPUT_TRANSFORMATION_FACTORY[
                extra_args["torch_model_architecture"]
            ]
            transformation = transformation_factory(meanstd)
        label_col = "Smiling"
        group_col = "Male"

        # Sample a subset of CelebA test split for the audit set
        rng = np.random.default_rng(auditset_seed)
        celeba = CelebADataset(
            target_columns=[label_col, group_col],
            transform=transformation,
            split="test",
        )
        indices = rng.choice(len(celeba), size=audit_pool_size, replace=False)
        celeba = torch.utils.data.Subset(celeba, indices=indices.tolist())

        features, [label, group] = load_whole_dataset(celeba)
        label = pd.Series(label)
        group = pd.Series(group).astype(bool)

        train_idx = np.array([])
        test_idx = np.array([])
        audit_idx = np.arange(audit_pool_size)
    else:
        raise NotImplementedError(f"The dataset {dataset} is not supported")

    print(f"dataset size: {features.shape[0]}")

    return features, label, group, train_idx, test_idx, audit_idx


def build_skorch_model(model_params: dict[str, Any]) -> scotch.NeuralNetClassifier:
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
        PretrainedFixedNetClassifier if frozen_params else scotch.NeuralNetClassifier
    )
    skorch_model = skorch_wrapper(
        module=architecture_factory,
        module__num_classes=num_classes,
    )
    return skorch_model


def generate_model(
    base_model_name: str,
    model_name: ModelName,
    model_params: dict[str, Any],
    strategy: ManipulationStrategy,
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


def detection_oracle(tpr: float, tnr: float):
    return AuditDetector(tpr, tnr)


def compute_metrics(
    X_queries: pd.DataFrame,
    y_queries: pd.Series,
    A_queries: pd.Series,
    y_pred: np.ndarray,
    y_pred_no_manipulation: np.ndarray,
    true_audit_queries_mask: np.ndarray,
) -> dict[str, float | list[float]]:
    """Compute different metrics related to the performance and fairness of the
    model."""

    metrics = {}
    groups = A_queries.unique()
    groups.sort()

    ############################################################################
    # SYSTEM (UNDER MANIPULATION) AS SEEN BY THE AUDITOR                       #
    ############################################################################
    # The utility as measured by the auditor
    metrics["utility_audit"] = accuracy_score(
        y_queries[true_audit_queries_mask], y_pred[true_audit_queries_mask]
    )

    # The per group conditional accuracy as seen by the users
    metrics["conditional_accuracy_audit"] = [
        accuracy_score(
            y_queries[true_audit_queries_mask],
            y_pred[true_audit_queries_mask],
            sample_weight=A_queries[true_audit_queries_mask] == group,
        )
        for group in groups
    ]

    # The demographic parity as seen by the auditor
    metrics["demographic_parity_audit"] = float(
        demographic_parity(
            y_queries[true_audit_queries_mask].to_numpy(),
            y_pred[true_audit_queries_mask],
            A_queries[true_audit_queries_mask].to_numpy(),
            mode="difference",
        )
    )

    # The demographic parity as seen by the auditor
    metrics["absolute_demographic_parity_audit"] = float(
        demographic_parity(
            y_queries[true_audit_queries_mask].to_numpy(),
            y_pred[true_audit_queries_mask],
            A_queries[true_audit_queries_mask].to_numpy(),
            mode="absolute_difference",
        )
    )

    ############################################################################
    # SYSTEM (NO MANIPULATION) AS SEEN BY THE AUDITOR                       #
    ############################################################################
    # The demographic parity with no manipulation
    metrics["demographic_parity_audit_honest"] = demographic_parity(
        y_queries[true_audit_queries_mask].to_numpy(),
        y_pred_no_manipulation[true_audit_queries_mask],
        A_queries[true_audit_queries_mask].to_numpy(),
        mode="difference",
    )
    # The absoulute demographic parity with no manipulation
    metrics["absolute_demographic_parity_audit_honest"] = demographic_parity(
        y_queries[true_audit_queries_mask].to_numpy(),
        y_pred_no_manipulation[true_audit_queries_mask],
        A_queries[true_audit_queries_mask].to_numpy(),
        mode="absolute_difference",
    )

    # The disagreement between orignal model and manipulations
    metrics["manipulation_hamming"] = np.mean(
        y_pred[true_audit_queries_mask]
        != y_pred_no_manipulation[true_audit_queries_mask]
    )

    ############################################################################
    # SYSTEM AS SEEN BY THE USERS                                              #
    ############################################################################
    # NOTE: The utility as seen by the users is disabled because it is never
    # used in the experiments for now.
    #
    # # The utility (for now, the accuracy) as seen by the users
    # metrics["utility_user"] = accuracy_score(
    #     y_queries[~true_audit_queries_mask], y_pred[~true_audit_queries_mask]
    # )

    # # The per group conditional accuracy as seen by the users
    # metrics["conditional_accuracy_user"] = [
    #     accuracy_score(
    #         y_queries[~true_audit_queries_mask],
    #         y_pred[~true_audit_queries_mask],
    #         sample_weight=A_queries[~true_audit_queries_mask] == group,
    #     )
    #     for group in groups
    # ]

    # # The demographic parity as seen by the users
    # metrics["demographic_parity_user"] = float(
    #     demographic_parity(
    #         y_queries[~true_audit_queries_mask].to_numpy(),
    #         y_pred[~true_audit_queries_mask],
    #         A_queries[~true_audit_queries_mask].to_numpy(),
    #         mode="difference",
    #     )
    # )
    # # The absolute demographic parity as seen by the users
    # metrics["absolute_demographic_parity_user"] = float(
    #     demographic_parity(
    #         y_queries[~true_audit_queries_mask].to_numpy(),
    #         y_pred[~true_audit_queries_mask],
    #         A_queries[~true_audit_queries_mask].to_numpy(),
    #         mode="absolute_difference",
    #     )
    # )

    return metrics


def run_audit(
    dataset: Dataset = "ACSEmployment",
    base_model_name: str = "skrub",
    model_name: ModelName = "unconstrained",
    model_params: str = "",
    training_imbalance: float | None = None,
    strategy: ManipulationStrategy = "honest",
    strategy_params: str = "",
    audit_budgets: int | list[int] = 1_000,
    audit_pool_size: int = 100,
    detection_tpr: float = 1.0,
    detection_tnr: float = 1.0,
    entropy: int = 123456789,
    override_seeds: dict[str, np.random.SeedSequence] | None = None,
    output: Path | None = None,
):
    """Run an audit simulation.

    1. Creates the audit/platform data separation
    2. Implements the platform model training
    3. Simulates the audit-points detection mechanism used by the platform
    4. Samples audit points and simulate the auditor querying the platform.
    5. Compute the true/estimated fairness/performance metrics and audit
       manipulation detection

    Parameters
    ----------

    training_imbalance: float, optional
        If not None, a percentage (equal to training_imbalance) of the training data with positive sensitive
        attribute will be removed to simulate biased training data. default: None
    """
    randomness = iter(np.random.SeedSequence(entropy).spawn(6))
    seeds = {
        "data_split": next(randomness),
        "train_test": next(randomness),
        "model": next(randomness),
        "audit_set": next(randomness),
        "audit_detector": next(randomness),
        "model_inference": next(randomness),
    }
    if override_seeds:
        seeds.update(override_seeds)
    if isinstance(audit_budgets, int):
        audit_budgets = [audit_budgets]

    ############################################################################
    # GENERATE THE DATA SPLITS                                                 #
    ############################################################################
    # Choose whether we want the sensitive feature to be binarized
    if "binarized" in dataset:
        dataset = dataset.replace("_binarized", "")
        binarize = True
    else:
        binarize = False

    # Extract the model params from the params string. For now, we assume that
    # there are only float params. Should be changed.
    model_params_dict = extract_params(model_params)
    strategy_params_dict = extract_params(strategy_params)

    extra_args = {}
    if dataset == "celeba" and base_model_name == "torch":
        assert (
            "model_architecture" in model_params_dict
        ), "The model architecture must be specified"
        extra_args = {
            "torch_model_architecture": model_params_dict["model_architecture"],
        }

    features, label, group, train_idx, test_idx, audit_idx = get_data(
        dataset, audit_pool_size, binarize, **extra_args
    )

    # Simulate a bias against one particular subgroup in the training data by
    # removing training samples with positive sensitive attribute.
    if training_imbalance is not None:
        train_idx_is_sensitive = (group.loc[train_idx] == True).to_numpy()
        n_train_sensitive = np.sum(train_idx_is_sensitive)
        train_idx_is_sensitive_subsampled = subsample_mask(
            train_idx_is_sensitive.copy(),
            num=int((1 - training_imbalance) * n_train_sensitive),
            seed=seeds["train_test"],
        )
        train_idx = train_idx[
            (~train_idx_is_sensitive) | train_idx_is_sensitive_subsampled
        ]

    X_train = get_subset(features, train_idx)
    y_train = label.loc[train_idx]
    A_train = group.loc[train_idx]

    X_test = get_subset(features, test_idx)
    y_test = label.loc[test_idx]
    A_test = group.loc[test_idx]

    assert (A_train.nunique() == group.nunique()) or len(
        train_idx
    ) == 0, "There are groups that are not represented in the train data"
    assert (group.loc[test_idx].nunique() == group.nunique()) or len(
        test_idx
    ) == 0, "There are groups that are not represented in the test data."
    # assert (
    #     len(features)
    #     == np.unique(np.concatenate([train_idx, test_idx, audit_idx])).shape[0]
    # ), "There are overlapping indices between the train, test and audit sets"

    ############################################################################
    # GENERATE AND TRAIN THE MANIPULATED MODEL                                 #
    ############################################################################
    model = generate_model(
        base_model_name,
        model_name,
        model_params_dict,
        strategy,
        strategy_params_dict,
        seeds["model"],
    )

    # Fit the model
    fit_time = perf_counter()
    model.fit(X_train, y_train, A_train)
    fit_time = perf_counter() - fit_time

    # Evaluate the perfs on train and test set (when not manipulated)
    if base_model_name != "torch":
        training_perfs = {
            "train_accuracy": np.mean(
                model.predict(
                    X_train,
                    A_train,
                    audit_queries_mask=np.zeros_like(y_train, dtype=bool),
                    random_state=random_state(seeds["model"]),
                )
                == y_train
            ),
            "test_accuracy": np.mean(
                model.predict(
                    X_test,
                    A_test,
                    audit_queries_mask=np.zeros_like(y_test, dtype=bool),
                    random_state=random_state(seeds["model"]),
                )
                == y_test
            ),
        }
    else:
        # TODO: save the train/test accuracies during celebA training and load
        # it back here
        training_perfs = {"train_accuracy": None, "test_accuracy": None}

    ############################################################################
    # GENERATE THE AUDIT DETECTION ORACLE                                      #
    ############################################################################
    oracle = detection_oracle(detection_tpr, detection_tnr)

    for audit_budget in audit_budgets:
        ############################################################################
        # RUN THE AUDIT SIMULATION                                                 #
        ############################################################################
        # Generate the audit set from the provided seed audit set
        if audit_budget < audit_pool_size:
            X_audit, y_audit, A_audit = audit_set(
                get_subset(features, audit_idx),
                label.loc[audit_idx],
                group.loc[audit_idx],
                audit_budget,
                seed=seeds["audit_set"],
            )

        # Compute all the audit metrics on the entire audit queries pool set.
        # This will be used as the ground truth values for all the metrics.
        elif audit_budget == audit_pool_size:
            X_audit, y_audit, A_audit = (
                get_subset(features, audit_idx),
                label.loc[audit_idx],
                group.loc[audit_idx],
            )

        else:
            raise ValueError(
                f"The {audit_budget = } must be less than the {audit_pool_size = }"
            )

        assert (
            A_audit.nunique() == group.nunique()
        ), "There are groups that are not represented in the audit seed data"

        # Generate the requests stream as seen by the platform (a.k.a. audit set + users set)
        if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
            X_queries = pd.concat([features.loc[test_idx], X_audit])
        elif isinstance(features, np.ndarray):
            X_queries = np.concatenate([features[test_idx], X_audit])
        elif isinstance(features, torch.Tensor):
            X_queries = torch.cat([features[test_idx], X_audit])
        y_queries = pd.concat([label.loc[test_idx], y_audit])
        A_queries = pd.concat([group.loc[test_idx], A_audit])

        # Simulate the audit queries detection mechanism
        true_audit_queries_mask = np.concatenate(
            [np.zeros(len(test_idx)), np.ones(audit_budget)]
        ).astype(bool)
        audit_queries_mask = oracle.detect(
            true_audit_queries_mask, seeds["audit_detector"]
        )

        # Ask the (potentially manipulated) model to label the queries
        inference_time = perf_counter()
        y_pred = model.predict(
            X_queries,
            A_queries,
            audit_queries_mask,
            random_state(seeds["model_inference"]),
        )
        inference_time = perf_counter() - inference_time

        # Ask the non manipulated values (aka predictions of the model if all
        # the queries are detected as non-audit)
        y_pred_no_manipulation = model.predict(
            X_queries,
            A_queries,
            np.zeros_like(audit_queries_mask, dtype=bool),
            random_state(seeds["model_inference"]),
        )

        ############################################################################
        # EVALUATE AND SAVE THE RESULTS                                            #
        ############################################################################
        record = dict(
            # Experiment parameters
            dataset=dataset,
            base_model_name=base_model_name,
            model_name=model_name,
            model_params=model_params_dict,
            training_imbalance=training_imbalance,
            strategy=strategy,
            strategy_params=strategy_params_dict,
            audit_budget=audit_budget,
            audit_pool_size=audit_pool_size,
            detection_tpr=detection_tpr,
            detection_tnr=detection_tnr,
            entropy=entropy,
            fit_time=fit_time,
            inference_time=inference_time,
            **training_perfs,
            **compute_metrics(
                X_queries,
                y_queries,
                A_queries,
                y_pred,
                y_pred_no_manipulation,
                true_audit_queries_mask,
            ),
        )

        if output:
            with output.open("a") as file:
                json.dump(record, file)
                file.write("\n")


@app.command()
def base_rates():
    """Compute the demographic parity on the entire dataset for all sensitive
    groups in the different states."""

    records = []
    groups = ["SEX", "RAC1P", "AGEP"]
    target = "ESR"
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")

    for year in range(2014, 2022):
        for sensitive_group, state in product(groups, state_list):
            data_source = ACSDataSource(
                survey_year=str(year), horizon="1-Year", survey="person"
            )
            acs_data = data_source.get_data(states=[state], download=True)
            ACSEmployment = folktables.BasicProblem(
                features=[
                    "AGEP",
                    "SCHL",
                    "MAR",
                    # "RELP", # Not present in 2019
                    "DIS",
                    "ESP",
                    "CIT",
                    "MIG",
                    "MIL",
                    "ANC",
                    "NATIVITY",
                    "DEAR",
                    "DEYE",
                    "DREM",
                    "SEX",
                    "RAC1P",
                ],
                target=target,
                target_transform=lambda x: x == 1,
                group=sensitive_group,
                preprocess=lambda x: x,
                # postprocess=lambda x: np.nan_to_num(x, -1),
            )
            features, labeldf, groupdf = ACSEmployment.df_to_pandas(acs_data)

            records.append(
                {
                    "year": year,
                    "state": state,
                    "len": len(features),
                    "group": sensitive_group,
                    "base_rate": float(
                        demographic_parity(
                            y_true=labeldf.to_numpy(),
                            y_pred=labeldf.to_numpy(),
                            A=groupdf.to_numpy(),
                        )
                    ),
                }
            )
            print(records[-1])

        # Delete the data source to save space
        list(
            map(lambda x: x.unlink(), (Path("./data") / str(year) / "1-Year").glob("*"))
        )

    pl.from_records(records).write_csv("test.csv")


@app.command()
def estimation_variance(run: bool = False):
    """Without manipulations, study the budget required to reduce variance and bias"""

    base_models = {"ACSEmployment_binarized": {"skrub", "skrub_logistic"}}
    AUDIT_POOL_SIZE = 10_000
    audit_budgets = [
        100,
        500,
        600,
        700,
        800,
        900,
        1_000,
        2_000,
        3_000,
        4_000,
        5_000,
        AUDIT_POOL_SIZE,
    ]
    n_repetitions = 15
    entropy = 12345678
    tnr = 1.0
    tpr = 1.0
    output_dir = Path(f"generated/estimation_variance-{n_repetitions}/")
    output_dir.mkdir(exist_ok=True, parents=True)

    if run:
        # Fix the randomness for everything except the audit_set selection
        seed = np.random.SeedSequence(entropy)
        override_seeds = {
            "data_split": seed.spawn(1)[0],
            "train_test": seed.spawn(1)[0],
            "model": seed.spawn(1)[0],
            "model_inference": seed.spawn(1)[0],
            "audit_detector": seed.spawn(1)[0],
            # "audit_set": seed.spawn(1)[0],
        }

        # Cleanup previous experiments
        for file in output_dir.glob("*.jsonl"):
            file.unlink()

        # Train and audit all the base models with no manipulation and no fair
        # training
        for dataset, base_model_names in base_models.items():
            for base_model_name, seed in product(
                base_model_names, np.random.SeedSequence(entropy).spawn(n_repetitions)
            ):
                output = output_dir / f"{dataset}_{base_model_name}.jsonl"

                print(f"{dataset = }, {base_model_name = }")
                run_audit(
                    dataset=dataset,
                    base_model_name=base_model_name,
                    model_name="unconstrained",
                    strategy="honest",
                    detection_tpr=tpr,
                    detection_tnr=tnr,
                    audit_budgets=audit_budgets,
                    audit_pool_size=AUDIT_POOL_SIZE,
                    entropy=int(random_state(seed)),
                    override_seeds=override_seeds,
                    output=output,
                )

    params = [
        "dataset",
        "base_model_name",
        # The fairness enhancing method, manipulation strategy and audit
        # detector are all the same so we comment them.
        #
        # "model_name", "model_params", "strategy", "strategy_params",
        # "detection_tpr", "detection_tnr",
        "audit_budget",
        # We average over the entropy, thus comment entropy.
        # "entropy",
    ]
    values = [
        "demographic_parity_audit",
        "demographic_parity_audit_honest",
        "entropy",
    ]

    # Compute the values for budget = audit pool size, to be used as the ground
    # truth
    reference = (
        pl.read_ndjson(list(output_dir.glob("*.jsonl")))
        .select(*params, *values)
        .filter(pl.col("audit_budget") == AUDIT_POOL_SIZE)
    )

    # Compute the bias and variance of the estimated value compared to the
    # ground truth
    estimation_variance = (
        pl.read_ndjson(list(output_dir.glob("*.jsonl")))
        .select(*params, *values)
        .filter(pl.col("audit_budget") < AUDIT_POOL_SIZE)
        .join(
            reference,
            on=["dataset", "base_model_name"],
            how="inner",
            suffix="_ref",
        )
        # Group by base_model_name and audit budget
        .group_by(params)
        # Compute the average and standart deviation over the differents seeds
        .agg(
            dp_audit_bias=(
                pl.col("demographic_parity_audit_honest")
                - pl.col("demographic_parity_audit_honest_ref")
            )
            .abs()
            .mean(),
            dp_audit_std=(
                pl.col("demographic_parity_audit_honest")
                - pl.col("demographic_parity_audit_honest_ref")
            )
            .abs()
            .std(),
        )
        # Compute the "error bars" and convert the struct columns into
        # jsonstring columns for plotly to be happy
        .with_columns(
            params_str=pl.concat_str(
                pl.col(params).exclude("audit_budget"), separator=","
            )
        )
        # Sort because plotly does not
        .sort("audit_budget")
    )

    fig = px.line(
        estimation_variance,
        x="audit_budget",
        y="dp_audit_bias",
        color="params_str",
        error_y="dp_audit_std",
        labels={
            "dp_audit_bias": r"$\mathbb{E}_S[|\hat{\mu}(S,h) - \mu(h)|]$",
            "audit_budget": r"$|S|$",
        },
        height=800,
    )
    fig.add_hline(
        reference["demographic_parity_audit_honest"].mean(),
        line_dash="dash",
        annotation_text="demographic parity (avg over models and runs)",
    )
    fig.update_traces(marker_size=20)
    fig.update_layout(
        title=dict(
            text="Bias and variance of the demographic parity estimation",
            subtitle=dict(
                text=f"y-axis value of each point is the bias over {n_repetitions} runs. The size of the error bars is the corresponding standard deviation.",
                font=dict(color="gray", size=13),
            ),
        )
    )
    fig.show()


@app.command()
def training_imbalance(run: bool = False):
    """Explore how much training imbalance is needed to significantly impact the
    demographic parity of the resulting model"""

    base_models = {"ACSEmployment_binarized": {"skrub", "skrub_logistic"}}
    AUDIT_POOL_SIZE = 10_000
    n_repetitions = 5
    entropy = 12345678
    tnr = 1.0
    tpr = 1.0
    training_imbalances = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    output_dir = Path(f"generated/training_imbalance-{n_repetitions}/")
    output_dir.mkdir(exist_ok=True, parents=True)

    if run:
        # Fix the randomness for everything except the audit_set selection
        seed = np.random.SeedSequence(entropy)
        override_seeds = {
            "data_split": seed.spawn(1)[0],
            # "train_test": seed.spawn(1)[0],
            "model": seed.spawn(1)[0],
            "model_inference": seed.spawn(1)[0],
            "audit_detector": seed.spawn(1)[0],
            "audit_set": seed.spawn(1)[0],
        }

        # Cleanup previous experiments
        for file in output_dir.glob("*.jsonl"):
            file.unlink()

        # Train and audit all the base models with no manipulation and no fair
        # training
        for dataset, base_model_names in base_models.items():
            for base_model_name, training_imbalance, seed in product(
                base_model_names,
                training_imbalances,
                np.random.SeedSequence(entropy).spawn(n_repetitions),
            ):
                output = (
                    output_dir
                    / f"{dataset}_{base_model_name}_{training_imbalance}.jsonl"
                )

                run_audit(
                    dataset=dataset,
                    base_model_name=base_model_name,
                    model_name="unconstrained",
                    training_imbalance=training_imbalance,
                    strategy="honest",
                    detection_tpr=tpr,
                    detection_tnr=tnr,
                    audit_budgets=AUDIT_POOL_SIZE,
                    audit_pool_size=AUDIT_POOL_SIZE,
                    entropy=int(random_state(seed)),
                    override_seeds=override_seeds,
                    output=output,
                )

    params = ["dataset", "base_model_name", "training_imbalance"]
    values = [
        "demographic_parity_audit_honest",
        "utility_audit",
        "train_accuracy",
        # "entropy",
    ]

    # Compute the values for budget = audit pool size
    training_info = (
        pl.read_ndjson(list(output_dir.glob("*.jsonl")))
        .filter(pl.col("audit_budget") == AUDIT_POOL_SIZE)
        .select(*params, *values)
        # Group by parameters (except budget since we look at the metrics
        # computed on the entire audit queries pool)
        .group_by("dataset", "base_model_name", "training_imbalance")
        # Compute the average over the different seeds
        .agg(
            pl.col(values).exclude("entropy").mean(),
            pl.col(values).exclude("entropy").std().name.suffix("_std"),
        )
        .unpivot(on=values, index=params)
    )
    training_info_std = (
        pl.read_ndjson(list(output_dir.glob("*.jsonl")))
        .filter(pl.col("audit_budget") == AUDIT_POOL_SIZE)
        .select(*params, *values)
        # Group by parameters (except budget since we look at the metrics
        # computed on the entire audit queries pool)
        .group_by("dataset", "base_model_name", "training_imbalance")
        # Compute the average over the different seeds
        .agg(
            pl.col(values).exclude("entropy").std().name.suffix("_std"),
        )
        .unpivot(
            on=[name + "_std" for name in values], index=params, value_name="value_std"
        )
        .with_columns(underlying_variable=pl.col("variable").str.replace("_std", ""))
    )
    training_info = (
        training_info.join(
            training_info_std,
            left_on=params + ["variable"],
            right_on=params + ["underlying_variable"],
        )
        .with_columns(
            params_str=pl.concat_str(
                pl.col(params).exclude("audit_budget", "training_imbalance"),
                separator=",",
            )
        )
        .sort("base_model_name", "training_imbalance")
    )
    print(training_info)

    fig = px.line(
        training_info,
        x="training_imbalance",
        y="value",
        facet_col="variable",
        error_y="value_std",
        color="params_str",
        markers=True,
        height=800,
    )
    # fig.update_traces(marker_size=20)
    fig.update_yaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.show()


@app.command()
def manipulation_stealthiness(run: bool = False):
    """Plot how much the un-fairness was lowered against how many points were
    changed"""

    audit_budget = 1_000
    # n_repetitions = 15
    n_repetitions = 5
    # n_repetitions = 1
    entropy = 12345678
    tnr = 1.0
    tpr = 1.0
    output = Path(f"generated/stealthiness{n_repetitions}.jsonl")

    base_models = {
        # "ACSEmployment_binarized": {
        #     "skrub", "skrub_logistic"
        # },
        "celeba": {
            "torch",
        }
    }

    model_params = {
        "ACSEmployment_binarized": "",
        "celeba": "model_architecture=lenet,num_classes=2,weight_path=data/models/lenet_celeba.pth",
    }

    if run:
        output.unlink(missing_ok=True)

        # Fix the randomness for everything except the audit_set selection
        seed = np.random.SeedSequence(entropy)
        override_seeds = {
            "data_split": seed.spawn(1)[0],
            "train_test": seed.spawn(1)[0],
            "model": seed.spawn(1)[0],
            "model_inference": seed.spawn(1)[0],
            "audit_detector": seed.spawn(1)[0],
            # "audit_set": seed.spawn(1)[0],
        }

        for dataset, dataset_base_models in base_models.items():
            print("Running experiments with dataset: ", dataset)
            for base_model, seed in product(
                dataset_base_models,
                np.random.SeedSequence(entropy).spawn(n_repetitions),
            ):
                print("honest unconstrained")
                run_audit(
                    dataset=dataset,
                    base_model_name=base_model,
                    model_name="unconstrained",
                    model_params=model_params[dataset],
                    strategy="honest",
                    detection_tpr=tpr,
                    detection_tnr=tnr,
                    audit_budgets=audit_budget,
                    audit_pool_size=10_000,
                    entropy=int(random_state(seed)),
                    override_seeds=override_seeds,
                    output=output,
                )

                # print("model swap")
                # run_audit(
                #     dataset=dataset,
                #     base_model_name=base_model,
                #     model_params=model_params[dataset],
                #     model_name="unconstrained",
                #     strategy="model_swap",
                #     detection_tpr=tpr,
                #     detection_tnr=tnr,
                #     audit_budgets=audit_budget,
                #     audit_pool_size=10_000,
                #     entropy=int(random_state(seed)),
                #     override_seeds=override_seeds,
                #     output=output,
                # )

                print("linear relaxation", end=" ", flush=True)
                for tolerated_unfairness in [0.0] + np.logspace(
                    -3, -1, num=10
                ).tolist():
                    print(f"{tolerated_unfairness:.3f}", end=" ", flush=True)
                    run_audit(
                        dataset=dataset,
                        base_model_name=base_model,
                        model_params=model_params[dataset],
                        model_name="unconstrained",
                        strategy="linear_relaxation",
                        strategy_params={"tolerated_unfairness": tolerated_unfairness},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=10_000,
                        entropy=int(random_state(seed)),
                        override_seeds=override_seeds,
                        output=output,
                    )
                print()

                print("score transport", end=" ", flush=True)
                for tolerated_unfairness in np.linspace(0, 1, num=10, endpoint=True):
                    print(f"{tolerated_unfairness:.3f}", end=" ", flush=True)
                    run_audit(
                        dataset=dataset,
                        base_model_name=base_model,
                        model_params=model_params[dataset],
                        model_name="unconstrained",
                        strategy="label_transport",
                        strategy_params={"tolerated_unfairness": tolerated_unfairness},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=10_000,
                        entropy=int(random_state(seed)),
                        override_seeds=override_seeds,
                        output=output,
                    )
                print()

                print("manipulation ROC", end=" ", flush=True)
                for theta in np.linspace(0.5, 0.6, num=10):
                    print(f"{theta:.2f}", end=" ", flush=True)
                    run_audit(
                        dataset=dataset,
                        base_model_name=base_model,
                        model_params=model_params[dataset],
                        model_name="unconstrained",
                        strategy="ROC_mitigation",
                        strategy_params={"theta": theta},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=10_000,
                        entropy=int(random_state(seed)),
                        override_seeds=override_seeds,
                        output=output,
                    )
                print()

    params = [
        "dataset",
        "base_model_name",
        "model_name",
        # "model_params",
        "strategy",
        "strategy_params",
        "audit_budget",
        # "detection_tpr",
        # "detection_tnr",
        # "entropy",
    ]

    model_perfs = (
        pl.read_ndjson(output)
        .select("train_accuracy", "entropy", *params)
        .filter(pl.col("train_accuracy").is_not_null())
        .sort("base_model_name")
    )
    print(model_perfs)
    # return

    records = (
        pl.read_ndjson(output)
        # For now, all model params are None so we can drop them
        .select(pl.all().exclude("model_params"))
        # The auditset hamming is what the auditor measures to detect manipulations
        .with_columns(
            auditset_hamming=1 - pl.col("utility_audit"),
            hidden_demographic_parity=(
                pl.col("demographic_parity_audit")
                - pl.col("demographic_parity_audit_honest")
            ).abs(),
            hidden_absolute_demographic_parity=(
                pl.col("absolute_demographic_parity_audit")
                - pl.col("absolute_demographic_parity_audit_honest")
            ).abs(),
        )
        # Group by all the parameters (model, manipulation and detection params)
        .group_by(params)
        # Compute the average and standart deviation over the differents seeds
        .agg(
            pl.all().mean(),
            pl.all().std().name.suffix("_std"),
        )
        # Compute the "error bars" and convert the struct columns into
        # jsonstring columns for plotly to be happy
        .with_columns(
            pl.col("demographic_parity_audit_std") / sqrt(n_repetitions),
            pl.col("hidden_demographic_parity_std") / sqrt(n_repetitions),
            pl.col("auditset_hamming_std") / sqrt(n_repetitions),
            pl.col("manipulation_hamming_std") / sqrt(n_repetitions),
            pl.col("strategy_params").struct.json_encode(),
        ).sort("strategy")
    )

    fig = px.scatter(
        records,
        x="auditset_hamming",
        y="hidden_demographic_parity",
        color="strategy",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        error_x="auditset_hamming_std",
        error_y="hidden_demographic_parity_std",
        hover_data=["strategy_params"],
    )
    fig.update_traces(marker_size=20)
    fig.show()

    fig = px.scatter(
        records,
        x="manipulation_hamming",
        y="hidden_demographic_parity",
        color="strategy",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        error_x="manipulation_hamming_std",
        error_y="hidden_demographic_parity_std",
        hover_data=["strategy_params"],
    )
    fig.update_traces(marker_size=20)
    fig.show()


@app.command()
def dev():
    from sklearn import set_config

    X, y, A = get_data("ACSIncome", binarize_group=False)
    model = generate_model(
        base_model_name="skrub",
        model_name="unconstrained",
        model_params={},
        strategy="honest",
        strategy_params={},
        seed=np.random.SeedSequence(123456789),
    )

    model = model.fit(X, y, sensitive_features=A)

    print(model.classes_)


@app.command()
def lenet():

    run_audit(
        dataset="celeba",
        base_model_name="torch",
        model_name="unconstrained",
        model_params="model_architecture=lenet,num_classes=2,weight_path=data/models/lenet_celeba.pth",
        # strategy="honest",
        strategy="model_swap",
        # strategy_params="tolerated_unfairness=0.1",
        # strategy_params="theta=0.55",
        # strategy_params="epsilon=0.1",
        audit_budgets=100,
        audit_pool_size=1_000,
        detection_tpr=1.0,
        detection_tnr=1.0,
        entropy=123456789,
        output=Path("./generated/abc.jsonl"),
        override_seeds=None,
    )

    #     strategy: ManipulationStrategy = "honest",
    #     strategy_params: str = "",
    #     audit_budget: int = 1_000,
    #     detection_tpr: float = 1.0,
    #     detection_tnr: float = 1.0,
    #     entropy: int = 123456789,
    #     override_seeds: dict[str, np.random.SeedSequence] | None = None,
    #     output: Path | None = None)


# app.command()(run_audit)


if __name__ == "__main__":
    app()
