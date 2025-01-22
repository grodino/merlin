from typing import Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
