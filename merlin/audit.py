from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from merlin.utils import random_state


def audit_set(
    features: pd.DataFrame | np.ndarray,
    label: pd.Series,
    group: pd.Series,
    budget: int,
    seed: np.random.SeedSequence,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X_audit, _, y_audit, _, A_audit, _ = train_test_split(
        features,
        label,
        group,
        train_size=budget,
        random_state=random_state(seed),
        # stratify=group,  # Select the same number of samples per group
    )

    return X_audit, y_audit, A_audit


def demographic_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    A: np.ndarray,
    mode: Literal["difference", "absolute_difference"] = "absolute_difference",
) -> float:
    """
    Computes demographic parity for a binary classification task.

    Parameters:
    -----------

    y_true: np.ndarray
        True labels.
    y_pred: np.ndarray
        Predicted labels.
    A: np.ndarray
        Protected attribute.
    """
    p_y1_a0 = np.mean(y_pred[A == 0])
    p_y1_a1 = np.mean(y_pred[A == 1])

    if mode == "difference":
        return (p_y1_a0 - p_y1_a1).item()
    elif mode == "absolute_difference":
        return np.abs(p_y1_a0 - p_y1_a1)
    else:
        raise ValueError("mode must be either 'difference' or 'absolute_difference'")


def performance_parity(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray):
    """
    Check whether the true and false positives are well distributed among the
    two sensitive groups.

    Introduced as FRAUD-detect in [Washing The Unwashable : On The
    (Im)possibility of Fairwashing
    Detection](https://openreview.net/forum?id=3vmKQUctNy).

    Parameters:
    -----------

    y_true: np.ndarray
        True labels.
    y_pred: np.ndarray
        Predicted labels.
    A: np.ndarray
        Protected attribute.
    """

    # True positives
    tp_a0 = np.sum((y_pred == 1) & (y_true == 1) & (A == 0)) / np.sum(
        (y_true == 1) & (A == 0)
    )
    tp_a1 = np.sum((y_pred == 1) & (y_true == 1) & (A == 1)) / np.sum(
        (y_true == 1) & (A == 1)
    )

    # False positives
    fp_a0 = np.sum((y_pred == 1) & (y_true == 0) & (A == 0)) / np.sum(
        (y_true == 0) & (A == 0)
    )
    fp_a1 = np.sum((y_pred == 1) & (y_true == 0) & (A == 1)) / np.sum(
        (y_true == 0) & (A == 1)
    )

    return np.abs(tp_a0 - tp_a1) + np.abs(fp_a0 - fp_a1)


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

    # The per group conditional accuracy as seen by the auditor
    metrics["conditional_accuracy_audit"] = [
        accuracy_score(
            y_queries[true_audit_queries_mask],
            y_pred[true_audit_queries_mask],
            sample_weight=A_queries[true_audit_queries_mask] == group,
        )
        for group in groups
    ]

    # The performance parity (a.k.a. FRAUD-detect) as seen by the auditor
    metrics["performance_parity_audit"] = float(
        performance_parity(
            y_queries[true_audit_queries_mask].to_numpy(),
            y_pred[true_audit_queries_mask],
            A_queries[true_audit_queries_mask].to_numpy(),
        )
    )

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

    # The performance parity (a.k.a. FRAUD-detect) as seen by the auditor
    metrics["performance_parity_audit_honest"] = float(
        performance_parity(
            y_queries[true_audit_queries_mask].to_numpy(),
            y_pred_no_manipulation[true_audit_queries_mask],
            A_queries[true_audit_queries_mask].to_numpy(),
        )
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
