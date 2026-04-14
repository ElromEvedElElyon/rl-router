"""Realistic example: route between OpenAI and Anthropic with fallback.

Uses fake providers so it runs standalone. Swap `fake_call_*` for real SDK
calls in production.
"""

from __future__ import annotations

import random
import time

from rl_router import Router


def fake_call_openai(prompt: str) -> str:
    time.sleep(random.uniform(0.2, 0.6))
    if random.random() < 0.1:
        raise RateLimitError("openai 429")
    return f"[openai]{prompt[:20]}"


def fake_call_anthropic(prompt: str) -> str:
    time.sleep(random.uniform(0.3, 0.9))
    if random.random() < 0.05:
        raise RateLimitError("anthropic 429")
    return f"[anthropic]{prompt[:20]}"


class RateLimitError(Exception):
    pass


PROVIDERS = {"openai": fake_call_openai, "anthropic": fake_call_anthropic}


def call_with_routing(router: Router, context, prompt: str, max_retries: int = 3):
    tried: set[str] = set()
    for _ in range(max_retries):
        arm = router.pick(context)
        if arm in tried:
            # Avoid retrying the same arm in the same call.
            remaining = [a for a in router.arms if a not in tried]
            if not remaining:
                break
            arm = remaining[0]
        tried.add(arm)

        t0 = time.time()
        try:
            out = PROVIDERS[arm](prompt)
        except RateLimitError:
            router.record(context, arm,
                          success=False,
                          latency_s=time.time() - t0,
                          rate_limited=True)
            continue
        router.record(context, arm,
                      success=True,
                      latency_s=time.time() - t0)
        return out, arm
    raise RuntimeError("all arms exhausted")


if __name__ == "__main__":
    r = Router(arms=list(PROVIDERS.keys()))
    counts = {"openai": 0, "anthropic": 0}
    for i in range(50):
        out, arm = call_with_routing(r, context=("en", "chat"),
                                     prompt=f"test prompt {i}")
        counts[arm] += 1
    print("Calls per arm:", counts)
    print(r.pretty())
