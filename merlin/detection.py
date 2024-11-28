import numpy as np


class AuditDetector:
    """An oracle that simulates how the platform tries to detect audit
    queries."""

    def __init__(self, tpr: float, tnr: float):
        self.tpr = tpr
        self.tnr = tnr

    def detect(
        self, audit_queries_mask: np.ndarray, seed: np.random.SeedSequence | None = None
    ) -> np.ndarray:

        rng = np.random.default_rng(seed)
        result = audit_queries_mask.copy()

        if (self.tpr, self.tnr) == (1.0, 1.0):
            return result

        result[audit_queries_mask == True] = rng.choice(
            [True, False],
            p=np.array([self.tpr, 1 - self.tpr]),
            size=np.sum(audit_queries_mask == True),
        )
        result[audit_queries_mask == False] = rng.choice(
            [True, False],
            p=np.array([1 - self.tnr, self.tnr]),
            size=np.sum(audit_queries_mask == False),
        )

        return result


def test_detector():
    """Test whether the observed true/false positive rates match the requested
    rates"""

    seed = np.random.SeedSequence(123456789)

    for tpr in np.linspace(0, 1, 10):
        for tnr in np.linspace(0, 1, 10):
            detector = AuditDetector(tpr, tnr)
            true_audit_queries_mask = np.concat(
                (np.ones(1_000, dtype=bool), np.zeros(20_000, dtype=bool))
            )
            audit_queries_mask = detector.detect(true_audit_queries_mask, seed)

            measured_tpr = np.mean(
                (audit_queries_mask == True)[true_audit_queries_mask == True]
            )
            measured_tnr = np.mean(
                (audit_queries_mask == False)[true_audit_queries_mask == False]
            )
            assert np.isclose(tpr, measured_tpr, atol=0.02)
            assert np.isclose(tnr, measured_tnr, atol=0.02)
