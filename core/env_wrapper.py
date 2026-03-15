"""Custom reward wrapper for Gymnasium MuJoCo environments."""

import gymnasium as gym
import numpy as np
from typing import Callable, Optional


class CustomRewardWrapper(gym.Wrapper):
    """Wraps a Gymnasium environment to use a custom reward function."""

    def __init__(self, env: gym.Env, reward_fn: Callable):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = original_reward

        try:
            custom_reward = float(self.reward_fn(self._prev_obs, action, obs, info))
        except Exception as e:
            custom_reward = original_reward
            info["reward_error"] = str(e)

        self._prev_obs = obs.copy()
        return obs, custom_reward, terminated, truncated, info


def make_env(
    env_id: str = "HalfCheetah-v5",
    reward_fn: Optional[Callable] = None,
    render: bool = False,
) -> gym.Env:
    """Create a MuJoCo environment.

    Args:
        render: If True, enable rgb_array rendering (for video). False for training.
    """
    kwargs = {}
    if render:
        kwargs["render_mode"] = "rgb_array"
    env = gym.make(env_id, **kwargs)
    if reward_fn is not None:
        env = CustomRewardWrapper(env, reward_fn)
    return env
