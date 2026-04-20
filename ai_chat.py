import os
from typing import List, Optional
from urllib.parse import quote

import requests


def _format_history_for_prompt(history: List[dict], user_prompt: str) -> str:
    """
    Pollinations no-key endpoint is simple prompt-in/text-out, so we serialize the chat.
    history items look like: {"role": "user"|"assistant", "content": "..."}.
    """
    lines: List[str] = []
    for m in history[-12:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    lines.append(f"User: {user_prompt.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def pollinations_chat(history: List[dict], user_prompt: str) -> str:
    """
    Uses Pollinations text endpoint that works without an API key.
    """
    prompt = _format_history_for_prompt(history, user_prompt)
    url = f"https://text.pollinations.ai/{quote(prompt)}"
    resp = requests.get(url, timeout=45)
    resp.raise_for_status()
    return resp.text.strip() or "Sorry, I didn't get a response. Please try again."


def openrouter_chat(history: List[dict], user_prompt: str) -> Optional[str]:
    """
    Optional fallback. Returns None when OPENROUTER_API_KEY is not set.
    """
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None

    messages = []
    for m in history[-20:]:
        role = m.get("role")
        if role not in ("user", "assistant", "system"):
            continue
        content = (m.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "messages": messages,
            "temperature": 0.3,
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def chat_reply(history: List[dict], user_prompt: str) -> str:
    """
    Pollinations-first (no key). If Pollinations fails, try OpenRouter if configured.
    """
    try:
        return pollinations_chat(history, user_prompt)
    except Exception as poll_err:
        fallback = openrouter_chat(history, user_prompt)
        if fallback:
            return fallback
        raise RuntimeError(
            f"Pollinations request failed and OpenRouter is not configured. ({poll_err})"
        )

