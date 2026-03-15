"""Evaluate trained agent and render video + LLM-readable report."""

import os
from typing import Callable, Optional

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

from core.env_wrapper import make_env

# HalfCheetah-v5 observation indices
OBS_LABELS = {
    0: "z_position (torso height)",
    1: "y_angle (torso angle)",
    2: "thigh_angle",
    3: "leg_angle",
    4: "foot_angle",
    5: "thigh_left_angle",
    6: "leg_left_angle",
    7: "foot_left_angle",
    8: "x_velocity (forward speed)",
    9: "z_velocity (vertical speed)",
    10: "y_angular_velocity (torso rotation speed)",
    11: "thigh_angular_velocity",
    12: "leg_angular_velocity",
    13: "foot_angular_velocity",
    14: "thigh_left_angular_velocity",
    15: "leg_left_angular_velocity",
    16: "foot_left_angular_velocity",
}

ACTION_LABELS = {
    0: "back_thigh",
    1: "back_shin",
    2: "back_foot",
    3: "front_thigh",
    4: "front_shin",
    5: "front_foot",
}


def evaluate_and_render(
    model: PPO,
    env_id: str = "HalfCheetah-v5",
    reward_fn: Optional[Callable] = None,
    output_dir: str = ".",
    run_name: str = "eval",
    n_episodes: int = 5,
    max_steps: int = 1000,
    training_npz_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate the model, render video, and save outputs.

    Outputs:
      - report.txt: Behavior + Physical Metrics + Episode Statistics
      - eval_data.npz: raw trajectories for programmatic use
      - eval.mp4: video for humans
    """
    os.makedirs(output_dir, exist_ok=True)

    env = make_env(env_id, reward_fn, render=True)

    frames = []
    episode_rewards = []
    episode_lengths = []
    all_obs = []
    all_actions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_obs = []
        ep_actions = []
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            if ep == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        if ep == 0:
            all_obs = np.array(ep_obs)
            all_actions = np.array(ep_actions)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    # --- Save video ---
    video_path = os.path.join(output_dir, "eval.mp4")
    if frames:
        imageio.mimsave(video_path, frames, fps=30)

    # --- Save report.txt ---
    report_text = _build_report_txt(
        env_id, episode_rewards, episode_lengths,
        all_obs, all_actions, training_npz_path,
    )
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # --- Save eval_data.npz (raw data) ---
    npz_data = {
        "episode_rewards": np.array(episode_rewards),
        "episode_lengths": np.array(episode_lengths),
        "obs_trajectory": all_obs,
        "action_trajectory": all_actions,
    }
    if training_npz_path and os.path.exists(training_npz_path):
        train_data = np.load(training_npz_path)
        npz_data["train_timesteps"] = train_data["timesteps"]
        npz_data["train_rewards"] = train_data["results"]
        npz_data["train_ep_lengths"] = train_data["ep_lengths"]

    npz_path = os.path.join(output_dir, "eval_data.npz")
    np.savez(npz_path, **npz_data)

    if verbose:
        print(report_text)

    # Compute physical metrics (unified, comparable across iterations)
    metrics = {}
    if len(all_obs) > 0:
        metrics["x_velocity"] = round(float(all_obs[:, 8].mean()), 3)
        metrics["z_velocity"] = round(float(all_obs[:, 9].mean()), 3)
        metrics["stability"] = round(float(all_obs[:, 1].std()), 3)
        metrics["height"] = round(float(all_obs[:, 0].mean()), 3)
        metrics["control_effort"] = round(float(np.mean(np.square(all_actions))), 4)

    stats = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "episodes": n_episodes,
        "video_path": video_path,
        "report_path": report_path,
        "npz_path": npz_path,
        "report": report_text,
        "metrics": metrics,
    }
    return stats


def _build_report_txt(
    env_id: str,
    episode_rewards: list,
    episode_lengths: list,
    obs_trajectory: np.ndarray,
    action_trajectory: np.ndarray,
    training_npz_path: Optional[str] = None,
) -> str:
    """Build report.txt with Behavior + Physical Metrics + Episode Statistics."""
    lines = []
    lines.append(f"EVALUATION REPORT: {env_id}")
    lines.append("=" * 50)

    # --- Section 1: Behavior ---
    lines.append("")
    lines.append("## Behavior")
    lines.append(_describe_behavior(obs_trajectory, action_trajectory, episode_rewards))

    # --- Section 2: Physical Metrics ---
    lines.append("")
    lines.append("## Physical Metrics")
    if len(obs_trajectory) > 0:
        x_velocity = float(obs_trajectory[:, 8].mean())
        z_velocity = float(obs_trajectory[:, 9].mean())
        stability = float(obs_trajectory[:, 1].std())
        height = float(obs_trajectory[:, 0].mean())
        control_effort = float(np.mean(np.square(action_trajectory)))
        lines.append(f"  x_velocity:      {x_velocity:>8.3f}")
        lines.append(f"  z_velocity:      {z_velocity:>8.3f}")
        lines.append(f"  stability:       {stability:>8.3f}")
        lines.append(f"  height:          {height:>8.3f}")
        lines.append(f"  control_effort:  {control_effort:>8.4f}")
    else:
        lines.append("  No observation data available.")

    # --- Section 3: Episode Statistics ---
    lines.append("")
    lines.append("## Episode Statistics")
    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    mean_len = np.mean(episode_lengths)
    lines.append(f"Episodes: {len(episode_rewards)}")
    lines.append(f"Mean Reward: {mean_r:.2f} (± {std_r:.2f})")
    lines.append(f"Mean Episode Length: {mean_len:.0f} / 1000 steps")
    lines.append(f"Reward Range: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")

    return "\n".join(lines)


def _describe_behavior(
    obs: np.ndarray,
    actions: np.ndarray,
    episode_rewards: list,
) -> str:
    if len(obs) == 0:
        return "No observation data available."

    descriptions = []

    if obs.shape[1] > 8:
        x_vel = obs[:, 8]
        avg_vel = x_vel.mean()
        if avg_vel > 2.0:
            descriptions.append(f"The agent runs forward at high speed (avg {avg_vel:.1f} m/s).")
        elif avg_vel > 0.5:
            descriptions.append(f"The agent moves forward at moderate speed (avg {avg_vel:.1f} m/s).")
        elif avg_vel > -0.5:
            descriptions.append(f"The agent barely moves (avg velocity {avg_vel:.2f} m/s).")
        else:
            descriptions.append(f"The agent moves backward (avg {avg_vel:.1f} m/s).")

    if obs.shape[1] > 1:
        angle = obs[:, 1]
        angle_std = angle.std()
        if angle_std > 0.5:
            descriptions.append("The torso is very unstable, wobbling significantly.")
        elif angle_std > 0.2:
            descriptions.append("The torso shows some wobble but is moderately stable.")
        else:
            descriptions.append("The torso is stable with minimal wobble.")

    if obs.shape[1] > 0:
        height = obs[:, 0]
        if height.min() < -0.5:
            descriptions.append("The agent falls down during the episode.")
        else:
            descriptions.append(f"The agent stays upright (height range: [{height.min():.2f}, {height.max():.2f}]).")

    effort = np.mean(np.square(actions))
    if effort > 0.5:
        descriptions.append("Control effort is high — large joint torques.")
    elif effort > 0.1:
        descriptions.append("Control effort is moderate.")
    else:
        descriptions.append("Control effort is low — mostly passive.")

    mean_r = np.mean(episode_rewards)
    if mean_r > 1000:
        descriptions.append("Overall: strong performance.")
    elif mean_r > 100:
        descriptions.append("Overall: learning in progress, showing some competence.")
    elif mean_r > 0:
        descriptions.append("Overall: minimal learning, barely performs the task.")
    else:
        descriptions.append("Overall: no meaningful learning yet, reward is negative.")

    return "\n".join(descriptions)
