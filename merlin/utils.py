from typing import Any

import numpy as np
from numpy.random import SeedSequence
from numpy.typing import ArrayLike


def random_state(seed: np.random.SeedSequence) -> np.int32:
    """Generates a random u32 from the given seed

    Required to adapt the new numpy rng system to the int-based system of
    sklearn.
    NOTE: Two calls with the same seed will return the same value.
    """
    return seed.generate_state(1)[0]


def extract_params(params_str: str | dict[str, Any]) -> dict[str, Any]:
    """Extract the model params from the params string "param1=0.1, param2=1"

    For now, we assume that there are only float params. Should be changed.
    """

    if isinstance(params_str, dict):
        return params_str

    params = {}

    for param in params_str.split(","):
        if "=" not in param:
            continue
        key, value = param.split("=")
        params[key.strip()] = float(value.strip())

    return params


def subsample_mask(
    mask: np.ndarray,
    num: int,
    seed: int | SeedSequence | None = None,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Only keep at most n true values (choosen at random) in the mask."""

    rng = np.random.default_rng(seed)
    # Number of positive mask values to remove
    to_remove = max(0, np.sum(mask) - num)

    # Choose `to_remove` positive mask values at random if no weights were given
    # or taking the samples with highest weight
    if weight is not None:
        # Sort the weights of the positive mask values in increasing order
        sorted_idx = np.argsort(weight[mask])

        # Remove the `to_remove` positive points with smallest weights from the mask
        remove = np.nonzero(mask)[0][sorted_idx[:to_remove]]
    else:
        remove = rng.choice(np.nonzero(mask)[0], size=to_remove, replace=False)

    # Set those mask values to 0
    mask[remove] = False

    return mask
