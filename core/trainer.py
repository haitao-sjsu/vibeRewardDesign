"""PPO training pipeline using stable-baselines3."""

import os
from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from core.env_wrapper import make_env


def train(
    env_id: str = "HalfCheetah-v5",
    reward_fn: Optional[Callable] = None,
    total_timesteps: int = 100_000,
    output_dir: str = "runs",
    run_name: str = "run",
    seed: int = 0,
    verbose: int = 1,
) -> PPO:
    """Train a PPO agent on the given environment.

    Returns the trained model.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Training envs: no rendering
    train_env = DummyVecEnv([lambda: make_env(env_id, reward_fn, render=False)])
    eval_env = DummyVecEnv([lambda: make_env(env_id, reward_fn, render=False)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(total_timesteps // 10, 1000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        seed=seed,
        device="auto",
    )

    if verbose:
        print(f"Training PPO on {env_id} for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model_path = os.path.join(run_dir, "final_model")
    model.save(model_path)
    if verbose:
        print(f"Model saved to {model_path}")

    train_env.close()
    eval_env.close()

    return model
