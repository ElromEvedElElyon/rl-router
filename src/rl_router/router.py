"""Contextual Thompson Sampling router for multi-provider LLM APIs.

Picks the best provider per context (e.g. per language, per task type) from
online feedback. Adapts to rate limits, quota resets, and latency drift
without manual tuning.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

# Portable file locking: fcntl on POSIX, msvcrt on Windows.
if sys.platform == "win32":
    import msvcrt

    def _lock_shared(fp):
        try:
            msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            pass

    def _unlock(fp):
        try:
            fp.seek(0)
            msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_shared(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_SH)

    def _unlock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


@dataclass
class RouterConfig:
    """Tunable knobs. Defaults are sensible for LLM provider routing."""

    half_life_calls: int = 500
    target_latency_s: float = 2.0
    rate_limit_penalty: float = 0.5
    exploration_floor: float = 0.02
    cooldown_s: float = 60.0
    cooldown_weight: float = 0.3


DEFAULT_STATE_PATH = Path(
    os.environ.get("RL_ROUTER_STATE", str(Path.home() / ".rl_router_state.json"))
)


def _decay(a: float, b: float, half_life: int, n_new: int = 1) -> tuple[float, float]:
    k = math.pow(0.5, n_new / half_life)
    return max(1.0, a * k), max(1.0, b * k)


class Router:
    """Multi-armed contextual bandit with Thompson Sampling.

    Example:
        >>> r = Router(arms=["openai", "anthropic", "local"])
        >>> arm = r.pick(context=("en", "summarization"))
        >>> # ... call provider[arm](prompt) ...
        >>> r.record(("en", "summarization"), arm,
        ...          success=True, latency_s=0.8)

    Thread/process-safe via atomic file replace + shared file lock on load.
    """

    def __init__(
        self,
        arms: Iterable[str],
        state_path: str | Path | None = None,
        config: RouterConfig | None = None,
    ):
        self.arms = list(arms)
        if not self.arms:
            raise ValueError("arms must not be empty")
        self.state_path = Path(state_path) if state_path else DEFAULT_STATE_PATH
        self.config = config or RouterConfig()
        self.state: dict = self._load()

    def _load(self) -> dict:
        if not self.state_path.exists():
            return {"ctx": {}, "calls": 0, "v": 1}
        try:
            with open(self.state_path, "r") as f:
                _lock_shared(f)
                try:
                    return json.load(f)
                finally:
                    _unlock(f)
        except (OSError, json.JSONDecodeError):
            return {"ctx": {}, "calls": 0, "v": 1}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            "w",
            dir=self.state_path.parent,
            delete=False,
            prefix=".rlr_",
            suffix=".tmp",
        )
        try:
            json.dump(self.state, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, self.state_path)
        except Exception:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise

    @staticmethod
    def _ctx_key(context: Hashable) -> str:
        if isinstance(context, str):
            return context
        if isinstance(context, (tuple, list)):
            return "|".join(map(str, context))
        return str(context)

    def _bucket(self, ck: str) -> dict:
        b = self.state["ctx"].setdefault(ck, {})
        for arm in self.arms:
            b.setdefault(
                arm,
                {"a": 1.0, "b": 1.0, "n": 0, "lat_ewma": 0.0, "last": 0.0, "last_rl": 0.0},
            )
        return b

    def pick(self, context: Hashable) -> str:
        """Return the arm to try next for this context."""
        ck = self._ctx_key(context)
        stats = self._bucket(ck)
        cfg = self.config

        if random.random() < cfg.exploration_floor:
            return random.choice(self.arms)

        best, best_score = self.arms[0], -math.inf
        now = time.time()
        for arm in self.arms:
            s = stats[arm]
            cooldown = (
                max(0.0, 1.0 - (now - s.get("last_rl", 0.0)) / cfg.cooldown_s)
                * cfg.cooldown_weight
            )
            sample = random.betavariate(s["a"], s["b"]) - cooldown
            if sample > best_score:
                best_score, best = sample, arm
        return best

    def record(
        self,
        context: Hashable,
        arm: str,
        *,
        success: bool,
        latency_s: float,
        rate_limited: bool = False,
    ) -> None:
        """Feed back the outcome of a call. Persists state to disk."""
        if arm not in self.arms:
            raise ValueError(f"unknown arm: {arm!r}")
        ck = self._ctx_key(context)
        stats = self._bucket(ck)
        s = stats[arm]
        cfg = self.config

        reward = 1.0 / (1.0 + latency_s / cfg.target_latency_s) if success else 0.0
        if rate_limited:
            reward -= cfg.rate_limit_penalty
            s["last_rl"] = time.time()

        s["a"], s["b"] = _decay(s["a"], s["b"], cfg.half_life_calls, n_new=1)

        r = max(-cfg.rate_limit_penalty, min(1.0, reward))
        span = 1.0 + cfg.rate_limit_penalty
        r01 = (r + cfg.rate_limit_penalty) / span
        s["a"] += r01
        s["b"] += 1.0 - r01

        alpha = 0.1
        s["lat_ewma"] = (1 - alpha) * s["lat_ewma"] + alpha * latency_s
        s["n"] += 1
        s["last"] = time.time()

        self.state["calls"] = self.state.get("calls", 0) + 1
        self._save()

    def stats(self) -> dict:
        out: dict = {}
        for ck, bucket in self.state["ctx"].items():
            out[ck] = {}
            for arm, s in bucket.items():
                total = s["a"] + s["b"]
                mean = s["a"] / total if total else 0.5
                var = (
                    (s["a"] * s["b"]) / (total * total * (total + 1))
                    if total > 1
                    else 0.25
                )
                out[ck][arm] = {
                    "n": s["n"],
                    "estimated_reward": round(mean, 3),
                    "uncertainty": round(math.sqrt(var), 3),
                    "latency_ewma_s": round(s["lat_ewma"], 2),
                }
        return out

    def pretty(self) -> str:
        buf = io.StringIO()
        for ck, arms in self.stats().items():
            buf.write(f"[{ck}]\n")
            ranked = sorted(
                arms.items(), key=lambda kv: -kv[1]["estimated_reward"]
            )
            for arm, m in ranked:
                buf.write(
                    f"  {arm:12s} r={m['estimated_reward']:.3f}"
                    f"±{m['uncertainty']:.3f}"
                    f"  n={m['n']:<5d} lat={m['latency_ewma_s']}s\n"
                )
        return buf.getvalue()
