import json
from itertools import product
from pathlib import Path
from typing import Annotated, Any
from functools import cache

import folktables
import numpy as np
import pandas as pd
import polars as pl
import sklearn as sk
import typer
from fairlearn.metrics._fairness_metrics import (
    demographic_parity_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    DemographicParity,
    ExponentiatedGradient,
)
from folktables import ACSDataSource, state_list
from plotly import express as px
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skrub import tabular_learner

from merlin.audit import audit_set
from merlin.detection import AuditDetector
from merlin.manipulation import (
    HonestClassifier,
    ModelSwap,
    RandomizedResponse,
    ROCMitigation,
)
from merlin.utils import extract_params, random_state


def plot():
    records = (
        pl.read_csv("generated/pareto.csv")
        .filter(pl.col("theta").is_null() | (pl.col("theta") < 0.6))
        .with_columns(
            param=pl.when(pl.col("model") == "exponentiated_gradient")
            .then("epsilon")
            .when(pl.col("model") == "tabular_learner")
            .then(pl.lit("tabular_learner"))
            .when(pl.col("model") == "threshold_optimizer")
            .then(pl.lit("threshold_optimizer"))
            .when(pl.col("model") == "manipulated_ROC")
            .then("theta")
            .when(pl.col("model") == "manipulated_rr")
            .then("epsilon")
        )
        .group_by("model", "param")
        .agg(
            pl.col("accuracy").mean(),
            pl.col("accuracy").std().name.suffix("_std"),
            pl.col("demographic_parity").mean(),
            pl.col("demographic_parity").std().name.suffix("_std"),
        )
        .with_columns(
            pl.col("demographic_parity_std") / np.sqrt(5),
            pl.col("accuracy_std") / np.sqrt(5),
        )
        .sort("model", "param")
    )

    # def hull(elements: pl.Series):
    #     fields = elements.struct.fields
    #     elements_arr = elements.struct.unnest().to_numpy()
    #     idx_hull = ConvexHull(elements_arr).vertices

    #     return (
    #         pl.Series(elements_arr[idx_hull, :])
    #         .cast(pl.List(pl.Float64))
    #         .list.to_struct(fields=fields)
    #         .explode()
    #     )

    # hull_points = (
    #     records.group_by("model", "param")
    #     .mean()
    #     .filter(pl.col("model").is_in(["ROC_mitigation", "exponentiated_gradient"]))
    #     .group_by("model")
    #     .agg(
    #         hull_points=pl.struct(["demographic_parity", "accuracy"]).map_elements(
    #             hull,
    #             return_dtype=pl.List(
    #                 pl.Struct(
    #                     fields=[
    #                         pl.Field("demographic_parity", pl.Float64),
    #                         pl.Field("accuracy", pl.Float64),
    #                     ]
    #                 )
    #             ),
    #         )
    #     )
    #     .filter(pl.col("hull_points").list.len() > 1)
    #     .explode("hull_points")
    #     .unnest("hull_points")
    # )

    fig = px.scatter(
        records,
        x="demographic_parity",
        y="accuracy",
        error_x="demographic_parity_std",
        error_y="accuracy_std",
        color="model",
        symbol="model",
        template="plotly_white",
        hover_data="param",
        width=1000,
        height=600,
    )
    fig.update_traces(marker_size=10)

    fig.show()


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
def get_data(dataset: Dataset, binarize_group: bool = False):
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    group_col = "AGEP"

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
        group: pd.Series = groupdf[group_col].astype(int)
        group.loc[groupdf[group_col] < 45] = 0
        group.loc[45 <= groupdf[group_col]] = 1
        group = group.astype(bool)

    else:
        raise NotImplementedError("Only support binary groups for now")

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

    return features, label, group


def generate_model(
    base_model_name: str,
    model_name: ModelName,
    model_params: dict[str, Any],
    strategy: ManipulationStrategy,
    strategy_params: dict[str, Any],
):

    match base_model_name:
        case "skrub_default" | "skrub":
            base_estimator = tabular_learner("classifier")
            sample_weight_name = "histgradientboostingclassifier__sample_weight"

        case "skrub_logistic":
            base_estimator = tabular_learner(LogisticRegression())
            sample_weight_name = "logisticregression__sample_weight"

        case _:
            raise NotImplementedError(
                f"The base base_model {base_model_name} is not supported"
            )

    match model_name:
        case "unconstrained":
            estimator = sk.clone(base_estimator)
            requires_sensitive_features = None

        case "exponentiated_gradient":
            epsilon = model_params["epsilon"]
            estimator = ExponentiatedGradient(
                sk.clone(base_estimator),
                DemographicParity(difference_bound=epsilon),
                sample_weight_name=sample_weight_name,
                eps=epsilon,
            )
            requires_sensitive_features = "fit"

        case "threshold_optimizer":
            estimator = ThresholdOptimizer(
                estimator=sk.clone(base_estimator), constraints="demographic_parity"
            )
            requires_sensitive_features = "both"

        case _:
            raise NotImplementedError(f"The model {model_name} is not supported")

    match strategy:
        case "honest":
            model = HonestClassifier(estimator, requires_sensitive_features)

        case "randomized_response":
            epsilon = strategy_params["epsilon"]
            model = RandomizedResponse(estimator, epsilon, requires_sensitive_features)

        case "ROC_mitigation":
            theta = strategy_params["theta"]
            model = ROCMitigation(estimator, theta, requires_sensitive_features)

        case "model_swap":
            model = ModelSwap(estimator, requires_sensitive_features)

        case _:
            raise NotImplementedError(
                f"The manipulation strategy {strategy} is not supported"
            )

    return model


def detection_oracle(tpr: float, tnr: float):
    return AuditDetector(tpr, tnr)


def compute_metrics(
    X_queries: pd.DataFrame,
    y_queries: pd.Series,
    A_queries: pd.Series,
    y_pred: np.ndarray,
    true_audit_queries_mask: np.ndarray,
) -> dict[str, float | list[float]]:
    metrics = {}
    groups = A_queries.unique()
    groups.sort()

    # The utility as seen by the users
    metrics["utility_user"] = accuracy_score(
        y_queries[~true_audit_queries_mask], y_pred[~true_audit_queries_mask]
    )
    # The utility as measured by the auditor
    metrics["utility_audit"] = accuracy_score(
        y_queries[true_audit_queries_mask], y_pred[true_audit_queries_mask]
    )
    # The per group conditional accuracy as seen by the users
    metrics["conditional_accuracy_user"] = [
        accuracy_score(
            y_queries[~true_audit_queries_mask],
            y_pred[~true_audit_queries_mask],
            sample_weight=A_queries[~true_audit_queries_mask] == group,
        )
        for group in groups
    ]

    # The demographic parity as seen by the users
    metrics["demographic_parity_user"] = float(
        demographic_parity_difference(
            y_queries[~true_audit_queries_mask],
            y_pred[~true_audit_queries_mask],
            sensitive_features=A_queries[~true_audit_queries_mask],
        )
    )
    # The demographic parity as seen by the auditor
    metrics["demographic_parity_audit"] = float(
        demographic_parity_difference(
            y_queries[true_audit_queries_mask],
            y_pred[true_audit_queries_mask],
            sensitive_features=A_queries[true_audit_queries_mask],
        )
    )
    # The per group conditional accuracy as seen by the auditor
    metrics["conditional_accuracy_user"] = [
        accuracy_score(
            y_queries[true_audit_queries_mask],
            y_pred[true_audit_queries_mask],
            sample_weight=A_queries[true_audit_queries_mask] == group,
        )
        for group in groups
    ]

    return metrics


def run_audit(
    dataset: Dataset = "ACSEmployment",
    base_model_name: str = "skrub",
    model_name: ModelName = "unconstrained",
    model_params: str = "",
    strategy: ManipulationStrategy = "honest",
    strategy_params: str = "",
    audit_budget: int = 1_000,
    detection_tpr: float = 1.0,
    detection_tnr: float = 1.0,
    entropy: int = 123456789,
    output: Path | None = None,
):
    AUDIT_SEED_SET_SIZE = 10_000
    seeds = np.random.SeedSequence(entropy).spawn(5)

    ############################################################################
    # GENERATE THE DATA SPLITS                                                 #
    ############################################################################
    # Choose whether we want the sensitive feature to be binarized
    if "binarized" in dataset:
        dataset = dataset.replace("_binarized", "")
        binarize = True
    else:
        binarize = False

    features, label, group = get_data(dataset, binarize)

    # The auditor has access to the same distribution as the model owner but
    # they do not have the exact same points
    train_test_idx, audit_idx = train_test_split(
        np.arange(len(features)),
        test_size=AUDIT_SEED_SET_SIZE,
        random_state=random_state(seeds[0]),
        stratify=group.astype(str) + "_" + label.astype(str),
    )

    # The model owner splits its data into train/test
    train_idx, test_idx = train_test_split(
        train_test_idx,
        test_size=0.3,
        random_state=random_state(seeds[1]),
        stratify=group.loc[train_test_idx].astype(str)
        + "_"
        + label.loc[train_test_idx].astype(str),
    )

    X_train = features.loc[train_idx]
    y_train = label.loc[train_idx]
    A_train = group.loc[train_idx]

    assert (
        A_train.nunique() == group.nunique()
    ), "There are groups that are not represented in the train data"
    assert (
        group.loc[test_idx].nunique() == group.nunique()
    ), "There are groups that are not represented in the train data"
    assert (
        len(features) == np.unique(np.concat([train_idx, test_idx, audit_idx])).shape[0]
    )

    ############################################################################
    # GENERATE AND TRAIN THE MANIPULMATED MODEL                                #
    ############################################################################
    # Extract the model params from the params string. For now, we assume that
    # there are only float params. Should be changed.
    model_params_dict = extract_params(model_params)
    strategy_params_dict = extract_params(strategy_params)

    model = generate_model(
        base_model_name, model_name, model_params_dict, strategy, strategy_params_dict
    )
    model.fit(X_train, y_train, A_train)

    ############################################################################
    # GENERATE THE AUDIT DETECTION ORACLE                                      #
    ############################################################################
    oracle = detection_oracle(detection_tpr, detection_tnr)

    ############################################################################
    # RUN THE AUDIT SIMULATION                                                 #
    ############################################################################
    # Generate the audit set from the provided seed audit set
    X_audit, y_audit, A_audit = audit_set(
        features.loc[audit_idx],
        label.loc[audit_idx],
        group.loc[audit_idx],
        audit_budget,
        seed=seeds[2],
    )
    assert (
        A_audit.nunique() == group.nunique()
    ), "There are groups that are not represented in the audit seed data"

    # Generate the requests stream as seen by the platform (a.k.a. audit set + users set)
    X_queries = pd.concat([features.loc[test_idx], X_audit])
    y_queries = pd.concat([label.loc[test_idx], y_audit])
    A_queries = pd.concat([group.loc[test_idx], A_audit])

    # Simulate the audit queries detection mechanism
    true_audit_queries_mask = np.concat(
        [np.zeros(len(test_idx)), np.ones(audit_budget)]
    ).astype(bool)
    audit_queries_mask = oracle.detect(true_audit_queries_mask, seeds[3])

    # Ask the (potentially manipulated) model to label the queries
    y_pred = model.predict(X_queries, A_queries, audit_queries_mask, seeds[4])

    ############################################################################
    # EVALUATE AND SAVE THE RESULTS                                            #
    ############################################################################
    record = dict(
        # Experiment parameters
        dataset=dataset,
        model_name=model_name,
        model_params=model_params_dict,
        strategy=strategy,
        strategy_params=strategy_params_dict,
        audit_budget=audit_budget,
        detection_tpr=detection_tpr,
        detection_tnr=detection_tnr,
        entropy=entropy,
        **compute_metrics(
            X_queries, y_queries, A_queries, y_pred, true_audit_queries_mask
        ),
    )

    if output:
        with output.open("a") as file:
            json.dump(record, file)
            file.write("\n")


@app.command()
def tradeoff():
    dataset = "ACSEmployment_binarized"
    base_model = "skrub_logistic"
    audit_budget = 1_000
    n_repetitions = 5
    entropy = 123456789
    output = Path(f"generated/dev{n_repetitions}_{dataset}_linear.jsonl")
    tpr = 1.0
    tnr = 1.0

    entropies = [
        random_state(seed)
        for seed in np.random.SeedSequence(entropy).spawn(n_repetitions)
    ]

    for entropy, (tpr, tnr) in product(
        entropies,
        ((0.95, 1.0), (1.0, 1.0), (0.5, 1.0)),
        # entropies, ((1.0, 1.0), (0.95, 1.0), (1.0, 0.95), (0.95, 0.95), (0.5, 1.0))
    ):
        print(entropy, tpr, tnr)
        print("unconstrained")
        run_audit(
            dataset=dataset,
            base_model_name=base_model,
            model_name="unconstrained",
            strategy="honest",
            detection_tpr=tpr,
            detection_tnr=tnr,
            audit_budget=audit_budget,
            entropy=int(entropy),
            output=output,
        )

        print("model swap")
        run_audit(
            dataset=dataset,
            base_model_name=base_model,
            model_name="unconstrained",
            strategy="model_swap",
            detection_tpr=tpr,
            detection_tnr=tnr,
            audit_budget=audit_budget,
            entropy=int(entropy),
            output=output,
        )

        print("manipulation randomized response", end=" ", flush=True)
        for epsilon in np.linspace(1, 10, num=10):
            print(f"{epsilon:.2f}", end=" ", flush=True)
            run_audit(
                dataset=dataset,
                base_model_name=base_model,
                model_name="unconstrained",
                strategy="randomized_response",
                strategy_params={"epsilon": epsilon},  # type: ignore
                detection_tpr=tpr,
                detection_tnr=tnr,
                audit_budget=audit_budget,
                entropy=int(entropy),
                output=output,
            )
        print()

        print("manipulation ROC", end=" ", flush=True)
        for theta in np.linspace(0.3, 0.6, num=10):
            print(f"{theta:.2f}", end=" ", flush=True)
            run_audit(
                dataset=dataset,
                base_model_name=base_model,
                model_name="unconstrained",
                strategy="ROC_mitigation",
                strategy_params={"theta": theta},  # type: ignore
                detection_tpr=tpr,
                detection_tnr=tnr,
                audit_budget=audit_budget,
                entropy=int(entropy),
                output=output,
            )
        print()

        print("fair model", end=" ", flush=True)
        for epsilon in np.linspace(0.001, 0.1, num=10):
            print(f"{epsilon:.2f}", end=" ", flush=True)

            run_audit(
                dataset=dataset,
                base_model_name=base_model,
                model_name="exponentiated_gradient",
                model_params={"epsilon": epsilon},  # type: ignore
                strategy="honest",
                detection_tpr=tpr,
                detection_tnr=tnr,
                audit_budget=audit_budget,
                entropy=int(entropy),
                output=output,
            )
        print()


@app.command()
def base_rates():
    records = []
    groups = ["SEX", "RAC1P", "AGEP"]
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
                target="ESR",
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
                        demographic_parity_difference(
                            y_true=labeldf, y_pred=labeldf, sensitive_features=groupdf
                        )
                    ),
                }
            )
            print(records[-1])

        list(
            map(lambda x: x.unlink(), (Path("./data") / str(year) / "1-Year").glob("*"))
        )

    pl.from_records(records).write_csv("test.csv")


@app.command()
def dev():
    dataset = "ACSEmployment_binarized"
    # base_model = "skrub_logistic"
    base_model = "skrub"
    audit_budget = 1_000
    n_repetitions = 5
    # n_repetitions = 1
    entropy = 123456789
    tnr = 1.0
    output = Path(f"generated/manipulation{n_repetitions}_{dataset}_{base_model}.jsonl")

    entropies = [
        random_state(seed)
        for seed in np.random.SeedSequence(entropy).spawn(n_repetitions)
    ]

    for entropy, tpr in product(entropies, np.linspace(0, 1.0, endpoint=True, num=5)):

        print("unconstrained")
        run_audit(
            dataset=dataset,
            base_model_name=base_model,
            model_name="unconstrained",
            strategy="honest",
            detection_tpr=tpr,
            detection_tnr=tnr,
            audit_budget=audit_budget,
            entropy=int(entropy),
            output=output,
        )

        print("model swap")
        run_audit(
            dataset=dataset,
            base_model_name=base_model,
            model_name="unconstrained",
            strategy="model_swap",
            detection_tpr=tpr,
            detection_tnr=tnr,
            audit_budget=audit_budget,
            entropy=int(entropy),
            output=output,
        )

        print("manipulation randomized response", end=" ", flush=True)
        for epsilon in np.linspace(1, 10, num=10):
            print(f"{epsilon:.2f}", end=" ", flush=True)
            run_audit(
                dataset=dataset,
                base_model_name=base_model,
                model_name="unconstrained",
                strategy="randomized_response",
                strategy_params={"epsilon": epsilon},  # type: ignore
                detection_tpr=tpr,
                detection_tnr=tnr,
                audit_budget=audit_budget,
                entropy=int(entropy),
                output=output,
            )
        print()

        print("manipulation ROC", end=" ", flush=True)
        for theta in np.linspace(0.3, 0.6, num=10):
            print(f"{theta:.2f}", end=" ", flush=True)
            run_audit(
                dataset=dataset,
                base_model_name=base_model,
                model_name="unconstrained",
                strategy="ROC_mitigation",
                strategy_params={"theta": theta},  # type: ignore
                detection_tpr=tpr,
                detection_tnr=tnr,
                audit_budget=audit_budget,
                entropy=int(entropy),
                output=output,
            )
        print()


app.command()(run_audit)


if __name__ == "__main__":
    app()
