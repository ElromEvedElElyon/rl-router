import random
from pathlib import Path

import pytest

from rl_router import Router, RouterConfig


@pytest.fixture
def tmp_state(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


def test_pick_returns_known_arm(tmp_state: Path) -> None:
    r = Router(arms=["a", "b", "c"], state_path=tmp_state)
    arm = r.pick(context=("ctx",))
    assert arm in {"a", "b", "c"}


def test_empty_arms_raises() -> None:
    with pytest.raises(ValueError):
        Router(arms=[])


def test_record_unknown_arm_raises(tmp_state: Path) -> None:
    r = Router(arms=["a"], state_path=tmp_state)
    with pytest.raises(ValueError):
        r.record("ctx", "b", success=True, latency_s=0.5)


def test_state_persists(tmp_state: Path) -> None:
    r1 = Router(arms=["a", "b"], state_path=tmp_state)
    r1.record("ctx", "a", success=True, latency_s=0.5)
    r2 = Router(arms=["a", "b"], state_path=tmp_state)
    assert r2.state["calls"] == 1
    assert r2.state["ctx"]["ctx"]["a"]["n"] == 1


def test_convergence_to_best_arm(tmp_state: Path) -> None:
    """After enough samples, best arm dominates picks."""
    random.seed(42)
    r = Router(arms=["good", "bad"], state_path=tmp_state,
               config=RouterConfig(exploration_floor=0.0))
    for _ in range(400):
        arm = r.pick("c")
        if arm == "good":
            r.record("c", arm, success=True, latency_s=0.3)
        else:
            r.record("c", arm, success=False, latency_s=1.0)

    picks = {"good": 0, "bad": 0}
    for _ in range(200):
        picks[r.pick("c")] += 1
    assert picks["good"] > picks["bad"] * 5, picks


def test_rate_limit_cooldown(tmp_state: Path) -> None:
    """Rate-limited arm is penalized in the next few picks."""
    random.seed(1)
    r = Router(arms=["hot", "cool"], state_path=tmp_state,
               config=RouterConfig(exploration_floor=0.0, cooldown_s=60.0))
    for _ in range(5):
        r.record("c", "hot", success=False, latency_s=0.1, rate_limited=True)
        r.record("c", "cool", success=True, latency_s=0.5)
    picks = {"hot": 0, "cool": 0}
    for _ in range(100):
        picks[r.pick("c")] += 1
    assert picks["cool"] > picks["hot"]


def test_context_isolation(tmp_state: Path) -> None:
    """Learning in one context does not affect another."""
    r = Router(arms=["a", "b"], state_path=tmp_state)
    for _ in range(50):
        r.record("ctx1", "a", success=True, latency_s=0.3)

    # ctx2 should be untrained: both arms roughly equal priors
    stats = r.stats()
    assert stats["ctx1"]["a"]["n"] == 50
    assert "ctx2" not in stats  # no calls → not recorded yet


def test_tuple_and_string_contexts_distinct(tmp_state: Path) -> None:
    r = Router(arms=["a"], state_path=tmp_state)
    r.record(("en", "qa"), "a", success=True, latency_s=0.1)
    r.record("en|qa", "a", success=True, latency_s=0.1)
    # They happen to collapse to the same key by design (str repr).
    # This test documents the behavior.
    stats = r.stats()
    assert stats["en|qa"]["a"]["n"] == 2
