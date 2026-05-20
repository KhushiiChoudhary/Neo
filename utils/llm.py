from __future__ import annotations

import os
import json
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def chat(system: str, user: str, model: str = "gpt-4o", temperature: float = 0.2) -> str:
    """Single-turn chat completion. Returns the assistant text."""
    response = get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def chat_json(system: str, user: str, model: str = "gpt-4o", temperature: float = 0.0) -> dict:
    """Chat completion that parses and returns a JSON object."""
    response = get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)
