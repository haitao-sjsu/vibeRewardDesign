# Vibe Reward Design

Use natural language to design reward functions for MuJoCo robots. An LLM generates the reward, PPO trains the policy, and the system iteratively improves based on evaluation results.

## How It Works

1. You describe a goal in plain language (e.g., "make the cheetah run backwards as fast as possible")
2. An LLM generates a Python reward function based on your goal
3. PPO trains a policy using that reward function
4. The system evaluates the trained agent and produces a report with physical metrics
5. The LLM reads the report and generates an improved reward function
6. Repeat for N iterations

## Quick Start

```bash
pip install -r requirements.txt
```

Create a `.env` file with your LLM provider config:

```
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1/
LLM_API_KEY=your_api_key
LLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
```

Run:

```bash
python vibe.py --goal "make the cheetah run backwards" --iterations 5 --steps 20000
```

## Output

Each session creates a directory under `runs/` with per-iteration results:

```
runs/vibe_14h52m45s/
  history.json       # full iteration history
  history.txt        # human-readable summary
  iter_0/
    reward_fn.py     # LLM-generated reward function
    report.txt       # evaluation report (behavior + metrics + stats)
    eval.mp4         # video of the trained agent
    eval_data.npz    # raw evaluation data
    train/           # PPO model checkpoints
  iter_1/ ...
```

## Physical Metrics

Instead of comparing raw reward values (which change with each reward function), the system tracks consistent physical metrics across iterations:

| Metric | Description |
|--------|-------------|
| x_velocity | Forward/backward speed (negative = backward) |
| z_velocity | Vertical speed (near 0 = stable) |
| stability | Torso angle std (lower = more stable) |
| height | Torso height (too low = fallen) |
| control_effort | Action magnitude (lower = more efficient) |

## Architecture

```
vibe.py                  # main loop: goal → LLM → train → evaluate → repeat
core/
  llm.py                 # LLM abstraction (Anthropic / OpenAI / Token Factory)
  reward_loader.py       # exec() generated reward code safely
  env_wrapper.py         # inject custom reward into Gymnasium env
  trainer.py             # PPO training via stable-baselines3
  evaluator.py           # evaluate agent, render video, generate report
prompts/
  system.txt             # system prompt with MuJoCo obs/action space docs
```

## Supported LLM Providers

Set `LLM_PROVIDER` in `.env`:

- `anthropic` — Claude (uses `ANTHROPIC_API_KEY`)
- `openai` — OpenAI or any OpenAI-compatible API (uses `LLM_BASE_URL` + `LLM_API_KEY`)

## Built At

[Nebius.Build SF Hackathon](https://cerebralvalley.ai/e/nebius-build-sf/details) — March 2026
