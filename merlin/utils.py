from typing import Any
import numpy as np


def random_state(seed: np.random.SeedSequence) -> np.int32:
    """Generates a random u32 from the given seed

    Required to adapt the new numpy rng system to the int-based system of
    sklearn.
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
