"""Synthetic benchmark: `python -m rl_router.simulate`.

Validates convergence against a known-truth oracle.
"""

from __future__ import annotations

import random
from pathlib import Path

from .router import Router


def main() -> None:
    random.seed(7)
    state = Path("/tmp/rl_router_sim.json") if Path("/tmp").exists() else Path.home() / ".rl_router_sim.json"
    if state.exists():
        state.unlink()
    r = Router(arms=["cerebras", "ollama", "gemini"], state_path=state)

    # Ground truth: (p_success, mean_latency_s) per (context, arm).
    table = {
        ("he", "torah"): {"cerebras": (0.95, 0.3), "ollama": (0.5, 4.0), "gemini": (0.7, 1.0)},
        ("en", "nt"):    {"cerebras": (0.85, 0.4), "ollama": (0.6, 3.5), "gemini": (0.9, 0.8)},
    }
    contexts = list(table.keys())

    for _ in range(2000):
        ctx = random.choice(contexts)
        arm = r.pick(ctx)
        p_ok, lat_mean = table[ctx][arm]
        ok = random.random() < p_ok
        lat = max(0.05, random.gauss(lat_mean, lat_mean * 0.3))
        rl = (not ok) and random.random() < 0.4
        r.record(ctx, arm, success=ok, latency_s=lat, rate_limited=rl)

    print("=== After 2000 iterations ===")
    print(r.pretty())

    picks = {ctx: {a: 0 for a in r.arms} for ctx in contexts}
    for _ in range(1000):
        ctx = random.choice(contexts)
        picks[ctx][r.pick(ctx)] += 1
    print("=== Post-training policy (1000 samples) ===")
    for ctx, p in picks.items():
        total = sum(p.values())
        row = ", ".join(f"{a}={c / total:.0%}" for a, c in p.items())
        print(f"  {ctx}: {row}")


if __name__ == "__main__":
    main()
