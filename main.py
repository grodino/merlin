import json
from itertools import product
from math import sqrt
from pathlib import Path
import re
from typing import Annotated
from time import perf_counter
import warnings

import folktables
import numpy as np
import pandas as pd
import polars as pl
import typer

import torch

from folktables import ACSDataSource, state_list
from plotly import express as px

from merlin.audit import audit_set, compute_metrics, demographic_parity
from merlin.detection import detection_oracle
from merlin.factory import generate_model
from merlin.utils import extract_params, get_subset, random_state, subsample_mask

from merlin.datasets import CelebADataset, get_data


app = typer.Typer()
ModelName = Annotated[str, "The name of the estimator trained by the platform"]
ManipulationStrategy = Annotated[
    str, "The strategy used by the platform to manipulate the answers on the audit set"
]
Dataset = Annotated[
    str,
    "The dataset on which the model is trained and frow which the audit set is sampled",
]


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
    # Extract the model params from the params string. For now, we assume that
    # there are only float params. Should be changed.
    model_params_dict = extract_params(model_params)
    strategy_params_dict = extract_params(strategy_params)

    # To know which data transform to use, we need to know which torch model is
    # used. (mainly for the data normalization.)
    extra_args = {}
    if dataset.startswith("celeba"):
        extra_args = {
            "torch_model_architecture": base_model_name,
        }

    features, label, group, train_idx, test_idx, audit_idx = get_data(
        dataset,
        audit_pool_size,
        traintest_seed=seeds["train_test"],
        auditset_seed=seeds["data_split"],
        **extra_args,
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

    # Fit the model (catch the warnings to avoid polluting the console)
    with warnings.catch_warnings(record=True) as train_warnings:
        fit_time = perf_counter()
        model.fit(X_train, y_train, A_train)
        fit_time = perf_counter() - fit_time

    # Evaluate the perfs on train and test set (when not manipulated)
    if not dataset.startswith("celeba"):
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

        # Ask the (potentially manipulated) model to label the queries (catch the warnings to avoid polluting the console)
        with warnings.catch_warnings(record=True) as inference_warnings:
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
            train_warnings=list(map(str, train_warnings)),
            inference_warnings=list(map(str, inference_warnings)),
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
def manipulation_stealthiness(run: bool = False, all_celeba_targets: bool = False):
    """Plot how much the un-fairness was lowered against how many points were
    changed"""

    audit_budget = [100, 1_000, 5_000]
    # n_repetitions = 15
    n_repetitions = 5
    # n_repetitions = 1
    entropy = 12345678
    tnr = 1.0
    tpr = 1.0
    audit_pool_size = 10_000
    output = Path(f"generated/stealthiness{n_repetitions}.jsonl")

    if not all_celeba_targets:
        celeba_targets = ["Smiling"]
    else:
        celeba_targets = CelebADataset.TRAINING_TARGETS

    base_models = (
        {
            f'celeba("{target}",gender,binarize_group=True)': [
                (
                    "lenet",
                    f'target="{target}",num_classes=2,weight_path=data/models/lenet/lenet_celeba_{target}.pth',
                )
            ]
            for target in celeba_targets
        }
        # | {
        #     f"resnet18": f"num_classes=2,weight_path=data/models/resnet18_celeba_{celeba_feature}.pth"
        #     for celeba_feature in celeba_targets
        # },
        # | {"ACSEmployment_binarized": [("skrub", ""), ("skrub_logistic", "")]}
    )

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
            print(f"{' Dataset: '+dataset+' ':-^81}")

            for (base_model, model_params), seed in product(
                dataset_base_models,
                np.random.SeedSequence(entropy).spawn(n_repetitions),
            ):
                print(
                    f"--- Base model: {base_model}({model_params:.60}{'...' if len(model_params) > 60 else ''}) [{seed.spawn_key}]"
                )
                print("    Manipulation: honest unconstrained")
                run_audit(
                    dataset=dataset,
                    base_model_name=base_model,
                    model_name="unconstrained",
                    model_params=model_params,
                    strategy="honest",
                    detection_tpr=tpr,
                    detection_tnr=tnr,
                    audit_budgets=audit_budget,
                    audit_pool_size=audit_pool_size,
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
                #     audit_pool_size=audit_pool_size,
                #     entropy=int(random_state(seed)),
                #     override_seeds=override_seeds,
                #     output=output,
                # )

                print("    Manipulation: threshold manipulation")
                run_audit(
                    dataset=dataset,
                    base_model_name=base_model,
                    model_params=model_params,
                    model_name="unconstrained",
                    strategy="threshold_manipulation",
                    detection_tpr=tpr,
                    detection_tnr=tnr,
                    audit_budgets=audit_budget,
                    audit_pool_size=audit_pool_size,
                    entropy=int(random_state(seed)),
                    override_seeds=override_seeds,
                    output=output,
                )

                print("    Manipulation: linear relaxation", end=" ", flush=True)
                for tolerated_unfairness in [0.0] + np.logspace(
                    -3, -1, num=10
                ).tolist():
                    print(f"{tolerated_unfairness:.3f}", end=" ", flush=True)
                    run_audit(
                        dataset=dataset,
                        base_model_name=base_model,
                        model_params=model_params,
                        model_name="unconstrained",
                        strategy="linear_relaxation",
                        strategy_params={"tolerated_unfairness": tolerated_unfairness},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=audit_pool_size,
                        entropy=int(random_state(seed)),
                        override_seeds=override_seeds,
                        output=output,
                    )
                print()

                print("    Manipulation: score transport", end=" ", flush=True)
                for tolerated_unfairness in np.linspace(0, 1, num=10, endpoint=True):
                    print(f"{tolerated_unfairness:.3f}", end=" ", flush=True)
                    run_audit(
                        dataset=dataset,
                        base_model_name=base_model,
                        model_params=model_params,
                        model_name="unconstrained",
                        strategy="label_transport",
                        strategy_params={"tolerated_unfairness": tolerated_unfairness},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=audit_pool_size,
                        entropy=int(random_state(seed)),
                        override_seeds=override_seeds,
                        output=output,
                    )
                print()

                print("    Manipulation: manipulation ROC", end=" ", flush=True)
                for theta in np.linspace(0.5, 0.6, num=10):
                    print(f"{theta:.2f}", end=" ", flush=True)
                    run_audit(
                        dataset=dataset,
                        base_model_name=base_model,
                        model_params=model_params,
                        model_name="unconstrained",
                        strategy="ROC_mitigation",
                        strategy_params={"theta": theta},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=audit_pool_size,
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
        .filter(pl.col("audit_budget") == 1_000)
        # .filter(pl.col("audit_budget").is_in([100, 1_000, 5_000]))
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
        )
        .sort("dataset", "audit_budget", "strategy")
    )

    fig = px.scatter(
        records,
        x="auditset_hamming",
        y="hidden_demographic_parity",
        color="strategy",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        facet_col="dataset",
        facet_row="audit_budget",
        symbol="base_model_name",
        error_x="auditset_hamming_std",
        error_y="hidden_demographic_parity_std",
        hover_data=["strategy_params"],
    )
    fig.update_traces(marker_size=20)
    # fig.update_xaxes(matches=None)
    fig.show()

    fig = px.scatter(
        records,
        x="manipulation_hamming",
        y="hidden_demographic_parity",
        color="strategy",
        symbol="base_model_name",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        facet_col="dataset",
        facet_row="audit_budget",
        error_x="manipulation_hamming_std",
        error_y="hidden_demographic_parity_std",
        hover_data=["strategy_params"],
    )
    # fig.update_xaxes(matches=None)
    fig.update_traces(marker_size=20)
    fig.show()

    fig = px.scatter(
        records,
        x="performance_parity_audit",
        y="hidden_demographic_parity",
        color="strategy",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        facet_col="dataset",
        facet_row="audit_budget",
        symbol="base_model_name",
        error_x="auditset_hamming_std",
        error_y="hidden_demographic_parity_std",
        hover_data=["strategy_params"],
    )
    fig.update_traces(marker_size=20)
    # fig.update_xaxes(matches=None)
    fig.show()

    fig = px.scatter(
        records,
        x="manipulation_hamming",
        y="performance_parity_audit",
        color="strategy",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        facet_col="dataset",
        facet_row="audit_budget",
        symbol="base_model_name",
        error_x="auditset_hamming_std",
        error_y="hidden_demographic_parity_std",
        hover_data=["strategy_params"],
    )
    fig.update_traces(marker_size=20)
    n_subplots = len([k for k in fig.layout if re.match(r"xaxis\d*$", k)])
    for i in range(1, n_subplots + 1):
        x0, y0, x1, y1 = 0.05, 0.05, 0.25, 0.25
        fig.add_shape(
            type="line",
            xref=f"x{i}",
            yref=f"y{i}",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="black", dash="dash"),
        )
    fig.update_traces(marker_size=20)
    # fig.update_xaxes(matches=None)
    fig.show()

    fig = px.scatter(
        records,
        x="auditset_hamming",
        y="performance_parity_audit",
        color="strategy",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        facet_col="dataset",
        facet_row="audit_budget",
        symbol="base_model_name",
        error_x="auditset_hamming_std",
        error_y="hidden_demographic_parity_std",
        hover_data=["strategy_params"],
    )
    fig.update_traces(marker_size=20)
    n_subplots = len([k for k in fig.layout if re.match(r"xaxis\d*$", k)])
    for i in range(1, n_subplots + 1):
        x0, y0, x1, y1 = 0.05, 0.05, 0.25, 0.25
        fig.add_shape(
            type="line",
            xref=f"x{i}",
            yref=f"y{i}",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="black", dash="dash"),
        )
    fig.update_traces(marker_size=20)
    # fig.update_xaxes(matches=None)
    fig.show()


@app.command()
def hiddable_unfairness(run: bool = False, all_celeba_targets: bool = False):
    """Plot how much unfairness can be hidden for given budget values and pre-specified thresholds"""

    audit_budget = [100, 300, 500, 700, 1_000, 2_000, 3_000, 4_000, 5_000]
    # n_repetitions = 15
    n_repetitions = 5
    # n_repetitions = 1
    entropy = 12345678
    tnr = 1.0
    tpr = 1.0
    audit_pool_size = 10_000
    output = Path(f"generated/stealthiness{n_repetitions}.jsonl")

    if not all_celeba_targets:
        celeba_targets = ["Smiling"]
    else:
        celeba_targets = CelebADataset.TRAINING_TARGETS

    base_models = {
        "celeba": {
            "lenet": f"target={celeba_feature},num_classes=2,weight_path=data/models/lenet/lenet_celeba_{celeba_feature}.pth"
            for celeba_feature in celeba_targets
        },
        # | {
        #     f"resnet18": f"num_classes=2,weight_path=data/models/resnet18_celeba_{celeba_feature}.pth"
        #     for celeba_feature in celeba_targets
        # },
        "ACSEmployment_binarized": {"skrub": "", "skrub_logistic": ""},
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

            for (base_model, model_params), seed in product(
                dataset_base_models.items(),
                np.random.SeedSequence(entropy).spawn(n_repetitions),
            ):
                print("honest unconstrained")
                run_audit(
                    dataset=dataset,
                    base_model_name=base_model,
                    model_name="unconstrained",
                    model_params=model_params,
                    strategy="honest",
                    detection_tpr=tpr,
                    detection_tnr=tnr,
                    audit_budgets=audit_budget,
                    audit_pool_size=audit_pool_size,
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
                #     audit_pool_size=audit_pool_size,
                #     entropy=int(random_state(seed)),
                #     override_seeds=override_seeds,
                #     output=output,
                # )

                print("threshold manipulation")
                run_audit(
                    dataset=dataset,
                    base_model_name=base_model,
                    model_params=model_params,
                    model_name="unconstrained",
                    strategy="threshold_manipulation",
                    detection_tpr=tpr,
                    detection_tnr=tnr,
                    audit_budgets=audit_budget,
                    audit_pool_size=audit_pool_size,
                    entropy=int(random_state(seed)),
                    override_seeds=override_seeds,
                    output=output,
                )

                print("linear relaxation", end=" ", flush=True)
                for tolerated_unfairness in [0.0] + np.logspace(
                    -3, -1, num=10
                ).tolist():
                    print(f"{tolerated_unfairness:.3f}", end=" ", flush=True)
                    run_audit(
                        dataset=dataset,
                        base_model_name=base_model,
                        model_params=model_params,
                        model_name="unconstrained",
                        strategy="linear_relaxation",
                        strategy_params={"tolerated_unfairness": tolerated_unfairness},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=audit_pool_size,
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
                        model_params=model_params,
                        model_name="unconstrained",
                        strategy="label_transport",
                        strategy_params={"tolerated_unfairness": tolerated_unfairness},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=audit_pool_size,
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
                        model_params=model_params,
                        model_name="unconstrained",
                        strategy="ROC_mitigation",
                        strategy_params={"theta": theta},  # type: ignore
                        detection_tpr=tpr,
                        detection_tnr=tnr,
                        audit_budgets=audit_budget,
                        audit_pool_size=audit_pool_size,
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
        "entropy",
    ]
    values = [
        "demographic_parity_audit",
        "demographic_parity_audit_honest",
        "absolute_demographic_parity_audit",
        "absolute_demographic_parity_audit_honest",
        "utility_audit",
        "performance_parity_audit",
        "manipulation_hamming",
    ]

    # NOTE: these were selected by hand, looking at the graphs from
    # `manipulation_stealthiness`. In practice, the threshold shoud also depend
    # on the type of model (or at least what a good accuracy is on this
    # dataset).
    detection_thresholds = pl.from_records(
        [
            {"dataset": "celeba", "threshold": 0.11, "base_model_name": "lenet"},
            {"dataset": "ACSEmployment", "threshold": 0.15, "base_model_name": "skrub"},
            {
                "dataset": "ACSEmployment",
                "threshold": 0.19,
                "base_model_name": "skrub_logistic",
            },
        ]
    )

    print(
        pl.read_ndjson(output)
        # For now, all model params are None so we can drop them
        .select(*params, *values)
        # Set the detection threshold for the different datasets and models
        # .join(detection_thresholds, on=["dataset"])
        .sample(10)
    )

    records = (
        pl.read_ndjson(output)
        # For now, all model params are None so we can drop them
        .select(*params, *values)
        # Set the detection threshold for the different datasets and models
        .join(detection_thresholds, on=["dataset", "base_model_name"])
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
        # Group by parameters except strategy and compute the max hidden unfairness
        .group_by(pl.col(params).exclude("strategy_params"))
        .agg(
            pl.col("hidden_demographic_parity")
            .filter(pl.col("auditset_hamming") < pl.col("threshold"))
            .max()
        )
        # Group by all the parameters (model, manipulation and detection params)
        .group_by(pl.col(params).exclude("entropy", "strategy_params"))
        # Compute the average and standart deviation over the differents seeds
        .agg(
            pl.all().mean(),
            pl.all().std().name.suffix("_std"),
        )
        # Compute the "error bars" and convert the struct columns into
        # jsonstring columns for plotly to be happy
        .with_columns(
            pl.col("hidden_demographic_parity_std") / sqrt(n_repetitions),
            # pl.col("auditset_hamming_std") / sqrt(n_repetitions),
            # pl.col("manipulation_hamming_std") / sqrt(n_repetitions),
            # pl.col("strategy_params").struct.json_encode(),
        )
        .sort("dataset", "audit_budget", "strategy")
    )

    fig = px.line(
        records,
        x="audit_budget",
        y="hidden_demographic_parity",
        error_y="hidden_demographic_parity_std",
        color="strategy",
        category_orders=dict(strategy=sorted(records["strategy"].sort().unique())),
        facet_col="dataset",
        symbol="base_model_name",
    )
    fig.update_traces(marker_size=20)
    # fig.update_xaxes(matches=None)
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
        strategy="threshold_manipulation",
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


@app.command()
def resnet():

    run_audit(
        dataset="celeba",
        base_model_name="resnet18",
        model_name="unconstrained",
        model_params="num_classes=2",
        # strategy="honest",
        strategy="honest",
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


if __name__ == "__main__":
    app()
