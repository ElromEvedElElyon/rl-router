"""Unit tests for the NVIDIA build.nvidia.com provider adapter.

All HTTP is mocked — no real key or network is required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import pytest

from rl_router import Router, RouterConfig
from rl_router.providers import NvidiaBuildProvider, NvidiaBuildResult


@pytest.fixture
def tmp_state(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


# ---------------------------------------------------------------------------
# Provider: request shape
# ---------------------------------------------------------------------------


def test_call_posts_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_post(url, body, headers):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(body.decode("utf-8"))
        payload = json.dumps(
            {"choices": [{"message": {"content": "pong"}}]}
        ).encode("utf-8")
        return 200, payload, ""

    p = NvidiaBuildProvider(api_key="test-key", model="meta/llama-3.1-405b-instruct")
    monkeypatch.setattr(p, "_http_post", fake_post)

    res = p.call("ping")

    assert res.success is True
    assert res.output == "pong"
    assert res.status_code == 200
    assert res.rate_limited is False
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "meta/llama-3.1-405b-instruct"
    assert captured["body"]["messages"][0]["content"] == "ping"


def test_is_available_requires_key() -> None:
    assert NvidiaBuildProvider(api_key="").is_available() is False
    assert NvidiaBuildProvider(api_key="abc").is_available() is True


def test_missing_key_returns_failed_result() -> None:
    p = NvidiaBuildProvider(api_key="")
    res = p.call("anything")
    assert res.success is False
    assert "NVIDIA_API_KEY" in res.error


# ---------------------------------------------------------------------------
# Rate-limit handling and cooldown cooperation with Router
# ---------------------------------------------------------------------------


def test_429_sets_rate_limited_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    p = NvidiaBuildProvider(api_key="k")
    monkeypatch.setattr(
        p, "_http_post", lambda url, body, headers: (429, b"", "429 Too Many Requests")
    )
    res = p.call("x")
    assert res.success is False
    assert res.rate_limited is True
    assert res.status_code == 429


def test_rate_limit_feeds_router_cooldown(
    tmp_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a rate-limited result, Router should penalise this arm via
    the cooldown term — so over many picks it loses to a healthy arm."""
    import random
    random.seed(1)
    cfg = RouterConfig(
        exploration_floor=0.0,
        cooldown_s=60.0,
        cooldown_weight=0.9,  # strong cooldown so the penalty is visible
    )
    r = Router(
        arms=["nvidia", "fallback"],
        state_path=tmp_state,
        config=cfg,
    )

    p = NvidiaBuildProvider(api_key="k")
    monkeypatch.setattr(
        p, "_http_post", lambda url, body, headers: (429, b"", "rate")
    )

    res = p.call("x")
    assert res.rate_limited is True
    r.record(
        ("en", "chat"),
        "nvidia",
        success=False,
        latency_s=res.latency_s,
        rate_limited=res.rate_limited,
    )

    # Immediately after: fallback should dominate picks while cooldown is hot.
    picks = {"nvidia": 0, "fallback": 0}
    for _ in range(200):
        picks[r.pick(("en", "chat"))] += 1
    assert picks["fallback"] > picks["nvidia"], picks


def test_embedded_quota_error_sets_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some providers wrap quota errors in HTTP 400 with a code field."""
    body = json.dumps(
        {"error": {"code": "rate_limit_exceeded", "message": "slow down"}}
    ).encode("utf-8")
    p = NvidiaBuildProvider(api_key="k")
    monkeypatch.setattr(p, "_http_post", lambda u, b, h: (400, body, "HTTP 400"))
    res = p.call("x")
    assert res.success is False
    assert res.rate_limited is True


# ---------------------------------------------------------------------------
# Malformed responses
# ---------------------------------------------------------------------------


def test_missing_choices_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    p = NvidiaBuildProvider(api_key="k")
    monkeypatch.setattr(
        p, "_http_post",
        lambda u, b, h: (200, json.dumps({"object": "nope"}).encode(), ""),
    )
    res = p.call("x")
    assert res.success is False
    assert "malformed" in res.error.lower()


def test_non_2xx_without_error_object(monkeypatch: pytest.MonkeyPatch) -> None:
    p = NvidiaBuildProvider(api_key="k")
    monkeypatch.setattr(
        p, "_http_post", lambda u, b, h: (503, b"<html>gateway</html>", "HTTPError 503: ..."),
    )
    res = p.call("x")
    assert res.success is False
    assert res.rate_limited is False
    assert res.status_code == 503
