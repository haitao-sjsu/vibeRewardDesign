"""LLM wrapper with provider abstraction (Anthropic / OpenAI-compatible)."""

import os
import re

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def complete(system_prompt: str, user_message: str) -> str:
    """Call LLM and return the response text."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    model = os.environ.get("LLM_MODEL", "")

    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_message, model or "claude-sonnet-4-20250514")
    else:
        return _call_openai(system_prompt, user_message, model or "gpt-4o")


def _call_anthropic(system_prompt: str, user_message: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _call_openai(system_prompt: str, user_message: str, model: str) -> str:
    import openai
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def extract_reward_code(response: str) -> str:
    """Extract Python code block from LLM response."""
    match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError("No code block found in LLM response")
