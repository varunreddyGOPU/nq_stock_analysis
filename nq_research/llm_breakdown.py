"""LLM-powered detailed breakdowns via local Ollama (default: glm-5.3-flash:cloud).

The engine computes; the LLM only NARRATES. Guardrails baked into the system
prompt: no predictions, always quote n/CI, label n<30 as anecdote, never
invent numbers. If Ollama is down, callers get the rule-based output anyway.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OLLAMA = os.environ.get("NQ_OLLAMA_URL", "http://localhost:11434")


def _model() -> str:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env.get("NQ_LLM_MODEL", os.environ.get("NQ_LLM_MODEL", "glm-5.3-flash:cloud"))


def ollama_chat(prompt: str, system: str, timeout: int = 120) -> str:
    """Chat completion via local Ollama; raises on transport errors."""
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={"model": _model(), "stream": False,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


SYSTEM = """You are the narration layer of a quantitative research engine that studies
historical base rates in the Nasdaq-100. You EXPLAIN computed statistics; you never
produce statistics yourself. Hard rules:
1. Never predict the market. Never say "will", "should buy", "bullish/bearish call".
2. Always quote the engine's n and confidence interval when present.
3. If n < 30 say the word "anecdote" explicitly.
4. Only use numbers given to you in the data block. If a number is missing, say so.
5. Explain in plain language what each metric means (ret, gap, VIX, term structure,
   OPEX, candle pattern, down-streak) for a smart retail trader.
6. End every reply with exactly: "History, not a signal."
Keep it under 180 words. Use Telegram HTML (<b>, <i>) sparingly."""


def explain_session(session_row: dict, forward: dict, base_rate_block: str) -> str:
    """Narrate one session card + its base-rate context."""
    payload = {
        "session": session_row,
        "forward_returns": forward,
        "base_rate_context": base_rate_block,
    }
    prompt = (
        "Data block (computed by the engine, trust it verbatim):\n"
        + json.dumps(payload, indent=1, default=str)
        + "\n\nTask: explain this session vividly but rigorously — what kind of day it was, "
        "what each number means, how the forward outcome compared with the base-rate block, "
        "and whether days like this historically resolved differently from the unconditional average."
    )
    return ollama_chat(prompt, SYSTEM)


def explain_result(result_report: str, condition: str) -> str:
    """Narrate a ConditionalResult.report() in plain language."""
    prompt = (
        f"Condition queried: {condition}\n\n"
        "Engine report (trust verbatim):\n" + result_report +
        "\n\nTask: explain what this conditional query found in plain English: the sample, "
        "the CI, whether it differs from the unconditional base rate, whether it is reliable "
        "at this n, and what a trader should NOT conclude from it."
    )
    return ollama_chat(prompt, SYSTEM)


def is_available() -> bool:
    try:
        return requests.get(f"{OLLAMA}/api/tags", timeout=4).ok
    except Exception:
        return False