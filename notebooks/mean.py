import numpy as np
from numba import njit


@njit
def block_mean(y: np.ndarray, n_blocks: int = 5, p: int = 2) -> float:
    (n_samples,) = y.shape
    group_size = n_samples // n_blocks

    group_mean = np.zeros(n_blocks)
    group_std = np.zeros(n_blocks)

    for i in range(n_blocks):
        group = y[i * group_size : (i + 1) * group_size]
        group_mean[i] = np.mean(group)
        group_std[i] = np.std(group)

    weights = np.zeros(n_blocks, dtype=np.float64)
    weights[group_std != 0] = 1 / group_std[group_std != 0] ** p
    weights = weights / np.sum(weights)

    return np.sum(weights * group_mean).item()


@njit
def mom(y: np.ndarray, n_groups: int = 5, n_permutations: int = 5) -> float:
    y = y.copy()
    (n_samples,) = y.shape
    group_size = n_samples // n_groups
    mom_value = 0

    for _ in range(n_permutations):
        group_mean = np.zeros(n_groups)
        np.random.shuffle(y)

        for i in range(n_groups):
            group = y[i * group_size : (i + 1) * group_size]

            group_mean[i] = np.mean(group)

        mom_value += np.median(group_mean).item()

    return mom_value / n_permutations
