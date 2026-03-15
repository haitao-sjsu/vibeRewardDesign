"""Load reward function from code string via exec()."""

import numpy as np


def load_reward_fn(code: str):
    """Execute code string and return the reward_fn callable.

    The code must define: def reward_fn(obs, action, next_obs, info) -> float
    numpy is available as np in the execution namespace.
    """
    namespace = {"np": np, "numpy": np}
    exec(code, namespace)
    if "reward_fn" not in namespace:
        raise ValueError("Code does not define 'reward_fn'")
    fn = namespace["reward_fn"]
    if not callable(fn):
        raise ValueError("'reward_fn' is not callable")
    return fn
