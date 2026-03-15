#!/usr/bin/env python3
"""Vibe RL — natural language driven reward iteration loop."""

import argparse
import json
import os
from datetime import datetime

from core.trainer import train
from core.evaluator import evaluate_and_render
from core.llm import complete, extract_reward_code
from core.reward_loader import load_reward_fn


def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system.txt")
    with open(prompt_path) as f:
        return f.read()


def build_user_message(goal: str, history: list) -> str:
    """Build the user message with goal + iteration history."""
    parts = [f'Goal: "{goal}"']

    if not history:
        parts.append("\nGenerate a reward function that achieves this goal.")
    else:
        parts.append("\nHere is what you've tried so far:\n")
        for entry in history[-3:]:
            parts.append(f"--- Iteration {entry['iteration']} ---")
            parts.append(f"Reward function:\n```python\n{entry['reward_code']}\n```")
            parts.append(f"Result:\n{entry['report']}")
            parts.append(f"Mean reward: {entry['mean_reward']:.2f}\n")
        parts.append("Analyze what worked and what didn't. Generate an improved reward function.")

    return "\n".join(parts)


def _write_history_txt(session_dir: str, goal: str, iterations: int, steps: int, history: list):
    """Write a human-readable history.txt summarizing the session so far."""
    lines = []
    lines.append("=== Vibe RL Session ===")
    lines.append(f"Goal: {goal}")
    lines.append(f"Iterations: {iterations} \u00d7 {steps} steps")
    lines.append("")

    for entry in history:
        lines.append(f"--- Iteration {entry['iteration']} (mean_reward: {entry['mean_reward']:.2f}) ---")
        lines.append("Reward function:")
        for code_line in entry["reward_code"].splitlines():
            lines.append(f"  {code_line}")
        lines.append("")

    lines.append("=== Reward Progression ===")
    for entry in history:
        lines.append(f"  Iter {entry['iteration']}: {entry['mean_reward']:.2f}")

    with open(os.path.join(session_dir, "history.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Vibe RL — LLM-driven reward iteration")
    parser.add_argument("--goal", required=True, help="Natural language goal")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--steps", type=int, default=10_000, help="Training timesteps per iteration")
    parser.add_argument("--env", default="HalfCheetah-v5", help="Gymnasium env ID")
    parser.add_argument("--output", default="runs", help="Output directory")
    args = parser.parse_args()

    # Create session directory
    timestamp = datetime.now().strftime("%Hh%Mm%Ss")
    session_name = f"vibe_{timestamp}"
    session_dir = os.path.join(args.output, session_name)
    os.makedirs(session_dir, exist_ok=True)

    system_prompt = load_system_prompt()
    history = []

    print(f"=== Vibe RL ===")
    print(f"Goal: {args.goal}")
    print(f"Iterations: {args.iterations} × {args.steps} steps")
    print(f"Output: {session_dir}/")
    print()

    for i in range(args.iterations):
        iter_dir = os.path.join(session_dir, f"iter_{i}")
        os.makedirs(iter_dir, exist_ok=True)

        # --- Step 1: LLM generates reward function ---
        print(f"--- Iteration {i} ---")
        print("Calling LLM to generate reward function...")
        user_message = build_user_message(args.goal, history)
        llm_response = complete(system_prompt, user_message)

        # Extract and save reward code
        max_retries = 3
        reward_code = None
        for attempt in range(max_retries):
            try:
                reward_code = extract_reward_code(llm_response)
                reward_fn = load_reward_fn(reward_code)
                break
            except Exception as e:
                print(f"  Error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    llm_response = complete(
                        system_prompt,
                        user_message + f"\n\nYour previous response had an error: {e}\nPlease fix it.",
                    )
                else:
                    print("  Failed to get valid reward function. Using default reward.")
                    reward_code = None
                    reward_fn = None

        # Save reward code
        if reward_code:
            with open(os.path.join(iter_dir, "reward_fn.py"), "w") as f:
                f.write(reward_code)
            print(f"  Reward function saved to {iter_dir}/reward_fn.py")

        # --- Step 2: Train ---
        print(f"  Training PPO for {args.steps} steps...")
        model = train(
            env_id=args.env,
            reward_fn=reward_fn,
            total_timesteps=args.steps,
            output_dir=iter_dir,
            run_name="train",
            seed=42 + i,
            verbose=0,
        )

        # --- Step 3: Evaluate ---
        train_dir = os.path.join(iter_dir, "train")
        npz_path = os.path.join(train_dir, "evaluations.npz")
        stats = evaluate_and_render(
            model=model,
            env_id=args.env,
            reward_fn=reward_fn,
            output_dir=iter_dir,
            run_name=f"{session_name}_iter{i}",
            training_npz_path=npz_path,
            verbose=False,
        )

        # --- Step 4: Record history ---
        m = stats.get("metrics", {})
        entry = {
            "iteration": i,
            "reward_code": reward_code or "(default reward)",
            "report": stats["report"],
            "mean_reward": stats["mean_reward"],
            "metrics": m,
        }
        history.append(entry)

        # Save history
        with open(os.path.join(session_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Save human-readable history.txt
        _write_history_txt(session_dir, args.goal, args.iterations, args.steps, history)

        print(f"  x_vel={m.get('x_velocity','?')}  z_vel={m.get('z_velocity','?')}  "
              f"stability={m.get('stability','?')}  height={m.get('height','?')}  "
              f"ctrl={m.get('control_effort','?')}")
        print()

    # --- Summary ---
    print("=== Summary ===")
    print(f"  {'Iter':<6} {'x_vel':>8} {'z_vel':>8} {'stabil':>8} {'height':>8} {'ctrl':>8}")
    for entry in history:
        m = entry.get("metrics", {})
        print(f"  {entry['iteration']:<6} {m.get('x_velocity','?'):>8} {m.get('z_velocity','?'):>8} "
              f"{m.get('stability','?'):>8} {m.get('height','?'):>8} {m.get('control_effort','?'):>8}")
    print(f"\nAll outputs in: {session_dir}/")


if __name__ == "__main__":
    main()
