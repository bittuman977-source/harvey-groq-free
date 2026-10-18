"""Tests for the Groq brain provider (mocked API — no key needed)."""
import asyncio
import json
import os
import sys

import httpx
import pytest

os.environ.setdefault("GROQ_API_KEY", "gsk_test_key")


def make_transport(status=200, content="hello from groq", usage=None,
                   body_override=None):
    usage = usage or {"prompt_tokens": 42, "completion_tokens": 7}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = body_override or {
            "model": "llama-3.3-70b-versatile",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage,
        }
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


# ---------- config validation ----------

def test_groq_is_accepted_provider():
    from harvey.config import BrainConfig, BRAIN_PROVIDERS
    assert "groq" in BRAIN_PROVIDERS
    cfg = BrainConfig(provider="groq")
    assert cfg.provider == "groq"
    cfg = BrainConfig(provider="groq", model="llama-3.3-70b-versatile")
    assert cfg.model == "llama-3.3-70b-versatile"


def test_bad_provider_still_rejected():
    from harvey.config import BrainConfig
    with pytest.raises(Exception):
        BrainConfig(provider="openai")


# ---------- GroqClient ----------

def test_client_defaults_and_key():
    from harvey.groq_client import GroqClient, DEFAULT_MODEL
    c = GroqClient(model="")
    assert c.model == DEFAULT_MODEL
    assert c.api_key() == "gsk_test_key"
    assert c.missing_key_reason() == ""


def test_missing_key_reason(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from harvey.groq_client import GroqClient
    c = GroqClient()
    assert "console.groq.com" in c.missing_key_reason()


@pytest.mark.asyncio
async def test_client_success():
    from harvey.groq_client import GroqClient
    c = GroqClient(transport=make_transport(content="SELL MORE"))
    result = await c.run("say something")
    assert result.text == "SELL MORE"
    assert result.usage["input_tokens"] == 42
    assert result.usage["output_tokens"] == 7
    assert result.usage["cost_usd"] == 0.0
    assert result.thread_id
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_client_rate_limit():
    from harvey.groq_client import GroqClient, GroqError
    c = GroqClient(transport=make_transport(status=429,
                body_override={"error": {"message": "Rate limit reached"}}))
    with pytest.raises(GroqError) as exc:
        await c.run("hi")
    assert exc.value.kind == "rate_limit"
    assert not exc.value.retryable


@pytest.mark.asyncio
async def test_client_auth_error():
    from harvey.groq_client import GroqClient, GroqError
    c = GroqClient(transport=make_transport(status=401,
                body_override={"error": {"message": "Invalid API Key"}}))
    with pytest.raises(GroqError) as exc:
        await c.run("hi")
    assert exc.value.kind == "auth"


@pytest.mark.asyncio
async def test_client_transient_500():
    from harvey.groq_client import GroqClient, GroqError
    c = GroqClient(transport=make_transport(status=503,
                body_override={"error": {"message": "unavailable"}}))
    with pytest.raises(GroqError) as exc:
        await c.run("hi")
    assert exc.value.kind == "transient"
    assert exc.value.retryable


# ---------- Brain integration (real Brain, mocked HTTP) ----------

class FakeState:
    def __init__(self):
        self.calls = 0
        self.events = []
        self.settings = {}

    async def increment_usage(self, provider="claude"):
        self.calls += 1

    async def record_usage_event(self, **kw):
        self.events.append(kw)

    async def get_setting(self, key):
        return self.settings.get(key)

    async def set_setting(self, key, value):
        self.settings[key] = value

    async def get_usage_today(self, provider="claude"):
        return float(self.calls)


def make_brain(transport):
    from harvey.brain import Brain
    from harvey.config import BrainConfig
    brain = Brain(FakeState(), BrainConfig(provider="groq"))
    brain.groq._transport = transport
    return brain


@pytest.mark.asyncio
async def test_brain_think_via_groq():
    brain = make_brain(make_transport(content="draft email here"))
    assert brain.provider == "groq"
    assert brain.provider_label == "Groq"
    preflight = await brain.preflight()
    assert preflight == ""
    text = await brain.think("write an email", session_id="harvey-writer")
    assert text == "draft email here"
    assert brain.state.calls == 1
    assert len(brain.state.events) == 1
    ev = brain.state.events[0]
    assert ev["provider"] == "groq"
    assert ev["source"] == "groq_api"
    assert ev["agent"] == "writer"  # derived from harvey-writer label
    assert ev["input_tokens"] == 42


@pytest.mark.asyncio
async def test_brain_preflight_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    brain = make_brain(make_transport())
    reason = await brain.preflight()
    assert "GROQ_API_KEY" in reason


@pytest.mark.asyncio
async def test_brain_rate_limit_pauses_budget():
    brain = make_brain(make_transport(status=429,
        body_override={"error": {"message": "Rate limit reached"}}))
    await brain.preflight()
    text = await brain.think("hello")
    assert text == ""
    assert brain.paused_reason  # paused, not silent failure
    within = await brain.is_within_budget()
    assert within is False  # gate closed during the pause


@pytest.mark.asyncio
async def test_brain_auth_error_marks_unavailable():
    brain = make_brain(make_transport(status=401,
        body_override={"error": {"message": "Invalid API Key"}}))
    await brain.preflight()
    text = await brain.think("hello")
    assert text == ""
    assert "API key" in brain.paused_reason.lower() or "rejected" in brain.paused_reason.lower()


@pytest.mark.asyncio
async def test_brain_budget_falls_back_to_call_count():
    brain = make_brain(make_transport())
    await brain.preflight()
    # No usage-limit pause, quota stub reports no windows -> call counter.
    assert await brain.is_within_budget(max_daily_calls=200) is True
    brain.state.calls = 500
    assert await brain.check_usage() == 500
