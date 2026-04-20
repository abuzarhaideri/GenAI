import os
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


def _split_keys(raw: str) -> List[str]:
    parts: List[str] = []
    for chunk in raw.replace(";", ",").replace("\n", ",").split(","):
        k = chunk.strip()
        if k:
            parts.append(k)
    return parts


def get_openai_keys_from_env() -> List[str]:
    raw_many = os.getenv("OPENAI_API_KEYS", "").strip()
    if raw_many:
        return _split_keys(raw_many)

    raw_one = os.getenv("OPENAI_API_KEY", "").strip()
    if raw_one and raw_one != "your_openai_api_key_here":
        return [raw_one]

    return []


def get_pollinations_key_from_env() -> Optional[str]:
    key = os.getenv("POLLINATIONS_API_KEY", "").strip()
    return key or None


@dataclass(frozen=True)
class ProviderConfig:
    pollinations_base_url: str = "https://gen.pollinations.ai/v1"
    pollinations_model: str = "openai"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.3
    timeout_s: float = 45.0
    pollinations_retry_once: bool = True


_rr_lock = threading.Lock()
_rr_index = 0


def _next_key(keys: Sequence[str]) -> str:
    global _rr_index
    if not keys:
        raise ValueError("No OpenAI keys available.")
    with _rr_lock:
        key = keys[_rr_index % len(keys)]
        _rr_index += 1
        return key


def _make_pollinations_llm(*, api_key: str, cfg: ProviderConfig) -> ChatOpenAI:
    # Pollinations is OpenAI-compatible via base_url.
    return ChatOpenAI(
        model=cfg.pollinations_model,
        api_key=api_key,
        base_url=cfg.pollinations_base_url,
        temperature=cfg.temperature,
        timeout=cfg.timeout_s,
        max_retries=0,
    )


def _make_openai_llm(*, api_key: str, cfg: ProviderConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.openai_model,
        api_key=api_key,
        temperature=cfg.temperature,
        timeout=cfg.timeout_s,
        max_retries=0,
    )


def invoke_with_pollinations_fallback(
    *,
    messages: Sequence[BaseMessage],
    tools: Iterable,
    cfg: Optional[ProviderConfig] = None,
):
    """
    Pollinations-first. If Pollinations errors/busy, retry once then fall back to OpenAI key rotation.
    """
    cfg = cfg or ProviderConfig()
    poll_key = get_pollinations_key_from_env()
    openai_keys = get_openai_keys_from_env()

    last_err: Optional[BaseException] = None

    if poll_key:
        try:
            llm = _make_pollinations_llm(api_key=poll_key, cfg=cfg).bind_tools(list(tools))
            return llm.invoke(list(messages))
        except BaseException as e:
            last_err = e
            if cfg.pollinations_retry_once:
                try:
                    llm = _make_pollinations_llm(api_key=poll_key, cfg=cfg).bind_tools(
                        list(tools)
                    )
                    return llm.invoke(list(messages))
                except BaseException as e2:
                    last_err = e2

    if openai_keys:
        key = _next_key(openai_keys)
        try:
            llm = _make_openai_llm(api_key=key, cfg=cfg).bind_tools(list(tools))
            return llm.invoke(list(messages))
        except BaseException as e:
            last_err = e

    if last_err is not None:
        raise RuntimeError(
            "No working AI provider available (Pollinations failed and OpenAI fallback unavailable/failed)."
        ) from last_err

    raise RuntimeError(
        "AI is not configured. Set POLLINATIONS_API_KEY (preferred) or OPENAI_API_KEYS."
    )

