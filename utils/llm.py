from __future__ import annotations

import os
import json
from openai import OpenAI

_client: OpenAI | None = None


def _get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in your .env file or Streamlit Cloud secrets."
    )


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=_get_api_key())
    return _client


def chat(system: str, user: str, model: str = "gpt-5.4", temperature: float = 0.2) -> str:
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


def chat_json(system: str, user: str, model: str = "gpt-5.4", temperature: float = 0.0) -> dict:
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
