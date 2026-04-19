"""Unit tests for the Claude CLI provider adapter and its RL Router wiring.

All tests mock :mod:`subprocess` — no real ``claude`` binary is required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from rl_router import Router, RouterConfig
from rl_router.providers import ClaudeCLIProvider, ClaudeCLIResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    """Mimic subprocess.CompletedProcess without invoking a real process."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Provider: command shape
# ---------------------------------------------------------------------------


def test_call_builds_expected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider must invoke `claude -p --max-turns 1 <prompt>`."""
    captured: dict = {}

    def fake_run(cmd, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        return _completed(0, stdout="pong\n")

    p = ClaudeCLIProvider(binary="/usr/bin/claude", timeout_s=30.0)
    monkeypatch.setattr(p, "_run", fake_run)

    res = p.call("ping")

    assert res.success is True
    assert res.output == "pong\n"
    assert res.return_code == 0
    assert res.cost_usd == 0.0
    assert captured["cmd"] == ["/usr/bin/claude", "-p", "--max-turns", "1", "ping"]
    assert captured["timeout"] == 30.0


def test_extra_args_are_customisable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run(cmd, timeout):
        captured["cmd"] = cmd
        return _completed(0, "ok")

    p = ClaudeCLIProvider(binary="claude", extra_args=("--verbose",))
    monkeypatch.setattr(p, "_run", fake_run)
    p.call("hello")

    assert captured["cmd"] == ["claude", "-p", "--verbose", "hello"]


# ---------------------------------------------------------------------------
# Provider: failure modes
# ---------------------------------------------------------------------------


def test_nonzero_exit_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    p = ClaudeCLIProvider(binary="claude")
    monkeypatch.setattr(
        p, "_run",
        lambda cmd, timeout: _completed(1, stdout="", stderr="auth error"),
    )
    res = p.call("x")
    assert res.success is False
    assert res.return_code == 1
    assert "auth error" in res.stderr
    assert res.timed_out is False


def test_timeout_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    p = ClaudeCLIProvider(binary="claude", timeout_s=0.5)

    def raising(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(p, "_run", raising)
    res = p.call("x")
    assert res.success is False
    assert res.timed_out is True
    assert res.return_code == -1


def test_binary_missing_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    p = ClaudeCLIProvider(binary="/nonexistent/claude")

    def raising(cmd, timeout):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(p, "_run", raising)
    res = p.call("x")
    assert res.success is False
    assert res.timed_out is False
    assert "not found" in res.stderr.lower()


# ---------------------------------------------------------------------------
# Provider: wiring into Router (Thompson Sampling updates)
# ---------------------------------------------------------------------------


def test_router_records_success(tmp_state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = Router(arms=["claude_cli", "fallback"], state_path=tmp_state)
    p = ClaudeCLIProvider(binary="claude")
    monkeypatch.setattr(p, "_run", lambda cmd, timeout: _completed(0, "hello"))

    res = p.call("prompt")
    r.record(("en", "chat"), "claude_cli",
             success=res.success, latency_s=res.latency_s)

    bucket = r.state["ctx"]["en|chat"]["claude_cli"]
    assert bucket["n"] == 1
    assert bucket["a"] > 1.0  # success bumps alpha


def test_router_records_failure(tmp_state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = Router(arms=["claude_cli", "fallback"], state_path=tmp_state)
    p = ClaudeCLIProvider(binary="claude")
    monkeypatch.setattr(
        p, "_run",
        lambda cmd, timeout: _completed(2, "", "quota exceeded"),
    )

    res = p.call("prompt")
    r.record(("en", "chat"), "claude_cli",
             success=res.success, latency_s=res.latency_s)

    bucket = r.state["ctx"]["en|chat"]["claude_cli"]
    assert bucket["n"] == 1
    # Failure: reward=0, so beta increments proportionally more than alpha.
    assert bucket["b"] > bucket["a"] - 0.5


# ---------------------------------------------------------------------------
# Router: prior bias so Claude CLI is preferred when available
# ---------------------------------------------------------------------------


def test_priors_biases_initial_pick(tmp_state: Path) -> None:
    """A strong prior on claude_cli makes it dominate before any data."""
    import random
    random.seed(0)
    r = Router(
        arms=["claude_cli", "cerebras", "ollama"],
        state_path=tmp_state,
        priors={"claude_cli": (10.0, 1.0)},
        config=RouterConfig(exploration_floor=0.0),
    )
    picks = {"claude_cli": 0, "cerebras": 0, "ollama": 0}
    for _ in range(200):
        picks[r.pick(("en", "chat"))] += 1
    assert picks["claude_cli"] > picks["cerebras"] + picks["ollama"], picks


def test_priors_only_seed_missing_contexts(tmp_state: Path) -> None:
    """Priors should not clobber learned state once calls have happened."""
    r = Router(
        arms=["claude_cli", "other"],
        state_path=tmp_state,
        priors={"claude_cli": (5.0, 1.0)},
    )
    # Record 20 failures for claude_cli — learned state should win over prior
    # shape after enough evidence.
    for _ in range(20):
        r.record(("ctx",), "claude_cli", success=False, latency_s=0.1)
    bucket = r.state["ctx"]["ctx"]["claude_cli"]
    # a stays near its decayed seed; b grows with failures.
    assert bucket["b"] > bucket["a"]


# ---------------------------------------------------------------------------
# Kill-switch: CLAUDE_CLI_DISABLED=1 forces the arm offline without editing
# code or removing the claude binary.
# ---------------------------------------------------------------------------


def test_kill_switch_env_var_marks_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CLI_DISABLED", "1")
    p = ClaudeCLIProvider(binary="/usr/bin/claude")
    assert p.is_available() is False


def test_kill_switch_short_circuits_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CLI_DISABLED", "1")

    called = {"ran": False}

    def tripwire(cmd, timeout):
        called["ran"] = True
        return _completed(0, "never")

    p = ClaudeCLIProvider(binary="/usr/bin/claude")
    monkeypatch.setattr(p, "_run", tripwire)

    res = p.call("ping")
    assert res.success is False
    assert called["ran"] is False
    assert "CLAUDE_CLI_DISABLED" in res.stderr


def test_kill_switch_empty_value_does_not_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CLI_DISABLED", "")
    p = ClaudeCLIProvider(binary="/usr/bin/claude")
    # Empty value is treated as unset (consistent with shell convention).
    monkeypatch.setattr(p, "_run", lambda cmd, timeout: _completed(0, "ok"))
    res = p.call("ping")
    assert res.success is True
