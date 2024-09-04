import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from merlin.utils import random_state


def audit_set(
    features: pd.DataFrame,
    label: pd.Series,
    group: pd.Series,
    budget: int,
    seed: np.random.SeedSequence,
):
    X_audit, _, y_audit, _, A_audit, _ = train_test_split(
        features,
        label,
        group,
        train_size=budget,
        random_state=random_state(seed),
        stratify=group,  # Select the same number of samples per group
    )

    return X_audit, y_audit, A_audit
