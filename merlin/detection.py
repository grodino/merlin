import numpy as np


class AuditDetector:
    """An oracle that simulates how the platform tries to detect audit
    queries."""

    def __init__(self, tpr: float, tnr: float):
        self.tpr = tpr
        self.tnr = tnr

    def detect(
        self, audit_queries_mask: np.ndarray, seed: np.random.SeedSequence | None
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # We assume the given audit queries mask is the ground truth
        true_negatives = audit_queries_mask == 0
        true_positives = audit_queries_mask == 1

        # Simulate the false positives.
        # We first create a binary mask that indicates which previously true
        # negatives are turned into false positives (resp. which previously true
        # positive are turned into false negatives)
        false_positives_mask = rng.choice(
            [False, True], p=[self.tpr, 1 - self.tpr], size=np.sum(true_negatives)
        )
        false_negatives_mask = rng.choice(
            [False, True], p=[self.tnr, 1 - self.tnr], size=np.sum(true_positives)
        )

        # Then, we flip the corresponding values
        audit_queries_mask[true_negatives][false_positives_mask] = 1
        audit_queries_mask[true_positives][false_negatives_mask] = 0

        return audit_queries_mask
