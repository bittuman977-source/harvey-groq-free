"""Groq brain client — free-tier thinking engine for Harvey.

Calls Groq's OpenAI-compatible chat completions API with a GROQ_API_KEY
(free at console.groq.com). Token accounting comes from the API response;
cost is recorded as 0 because the free tier bills nothing. No tools are
ever sent: prompts carry untrusted text (prospect replies, scraped
pages), and Harvey only ever needs text back.
"""

import asyncio
import logging
import os
import time
import uuid

import httpx

logger = logging.getLogger("harvey.groq")

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

# status codes: 429 = rate limit (respects a pause), 401/403 = bad key
# (needs operator action), 5xx/other = transient (retry).
_AUTH_KINDS = {401: "auth", 403: "auth", 429: "rate_limit"}


class GroqError(Exception):
    """One failed Groq call. ``kind`` is "auth" (operator must fix the key),
    "rate_limit" (free-tier limit hit; pause and wait), or "transient"."""

    def __init__(self, message: str, kind: str = "transient", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class GroqResult:
    """Text plus accounting for one completed Groq call."""

    def __init__(self, text: str, model: str, thread_id: str, duration_ms: int,
                 usage: dict):
        self.text = text
        self.model = model
        self.thread_id = thread_id
        self.duration_ms = duration_ms
        self.usage = usage


def _normalize_usage(response_usage) -> dict:
    """Groq's token counts -> Harvey's usage_events columns."""
    if not isinstance(response_usage, dict):
        return {}
    prompt = response_usage.get("prompt_tokens") or 0
    completion = response_usage.get("completion_tokens") or 0
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "cost_usd": 0.0,  # free tier: no bill, so no estimated cost
    }


class GroqClient:
    """Async chat-completion client for the Groq API."""

    def __init__(self, model: str = "", base_url: str | None = None,
                 api_key: str | None = None, transport=None):
        self.model = model or DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("GROQ_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key  # None -> read GROQ_API_KEY at call time
        self._transport = transport  # test injection only

    def api_key(self) -> str:
        return self._api_key if self._api_key is not None else os.environ.get(
            "GROQ_API_KEY", "").strip()

    def missing_key_reason(self) -> str:
        if self.api_key():
            return ""
        return ("GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys, add it to .env, and restart "
                "Harvey.")

    async def run(self, prompt: str, timeout: float = 120.0) -> GroqResult:
        """One chat completion. Raises GroqError on failure."""
        api_key = self.api_key()
        if not api_key:
            raise GroqError(self.missing_key_reason(), kind="auth")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # Prompts are large (skills + scraped pages); temperature 0.7
            # matches Harvey's writer defaults well enough for free models.
            "temperature": 0.7,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            raise GroqError(f"Groq call timed out: {e}", kind="transient") from e
        except httpx.HTTPError as e:
            raise GroqError(
                f"Groq network error: {type(e).__name__}", kind="transient"
            ) from e

        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                detail = str(body.get("error", {}).get("message", ""))[:200]
            except Exception:
                detail = response.text[:200]
            kind = _AUTH_KINDS.get(response.status_code, "transient")
            message = f"Groq API error {response.status_code}: {detail}"
            if kind == "rate_limit":
                message = f"Groq free-tier rate limit hit: {detail or response.status_code}"
            elif kind == "auth":
                message = f"Groq rejected the API key: {detail or response.status_code}"
            raise GroqError(message, kind=kind, retryable=(kind == "transient"))

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
            model = data.get("model", self.model)
            usage = _normalize_usage(data.get("usage"))
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise GroqError(
                f"Groq returned an unreadable response: {e}", kind="transient"
            ) from e

        duration_ms = int((time.monotonic() - started) * 1000)
        return GroqResult(
            text=text, model=model, thread_id=str(uuid.uuid4()),
            duration_ms=duration_ms, usage=usage,
        )
