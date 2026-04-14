# rl-router

**Contextual Thompson Sampling router for multi-provider LLM APIs.**
Zero config, zero hyperparameters to tune, adapts online to rate limits, quota resets, and latency drift.

```python
from rl_router import Router

r = Router(arms=["openai", "anthropic", "cerebras", "local"])

arm = r.pick(context=("en", "summarization"))
t0 = time.time()
try:
    out = call_provider[arm](prompt)
    r.record(("en", "summarization"), arm,
             success=True, latency_s=time.time()-t0)
except RateLimitError:
    r.record(("en", "summarization"), arm,
             success=False, latency_s=time.time()-t0, rate_limited=True)
```

That's the whole API. The router learns which provider wins per context, *per context*.

---

## Why not just try-catch a provider list?

Hand-written fallback logic fails silently in three common ways:

| Failure mode | Static fallback | rl-router |
|---|---|---|
| Provider A hits rate limit mid-job | Keeps hammering A, retries noisy | Shifts weight to B in real time, cools A down 60s |
| Quota resets at midnight UTC | Never recovers A until you redeploy | Decay (half-life 500 calls) re-explores A automatically |
| Latency degrades on B | No signal, users complain | Reward penalizes slow success, picks C |
| Workers run in parallel (saturation) | All hit same provider, worsen 429 | Thompson sampling diversifies independently |

Bandits aren't new. What's new is **wiring them to the LLM ops problem with zero config** and surviving real production pain (file-locked state, atomic writes, decay, cooldown).

## Install

```
pip install rl-router
```

No dependencies. Python 3.9+. Works on Linux, macOS, Windows.

## Core concepts (30 seconds)

- **Arms** = providers you can call (`openai`, `anthropic`, `local`, whatever).
- **Context** = any hashable tuple describing the task (`(language, task_type)`, `(user_tier, model_size)`, etc). The router learns a *separate* policy per context.
- **Reward** = `success × 1/(1+latency/2s) − 0.5×rate_limited`. Fast successes beat slow successes; 429s hurt extra.
- **Beta(α,β)** per (context, arm) encodes belief about that arm's reward.
- **Thompson sampling** = sample from each Beta, pick the max. Naturally balances explore/exploit.
- **Decay** (half-life 500 calls) ages old evidence so the router reacts to quota resets and service degradation.

## Why Thompson Sampling (not ε-greedy, not UCB)

- **vs ε-greedy**: doesn't waste ε on known-bad arms.
- **vs UCB**: deterministic → parallel workers pick the *same* arm and collapse the same provider. Thompson samples → parallel workers naturally spread.
- **vs hand-tuned weights**: no weights to tune. Bayesian prior → posterior does the work.

## Benchmark (synthetic, reproducible)

`python -m rl_router.simulate` runs a 2000-step sim against a known ground truth:

```
[he|torah]
  cerebras     r=0.880±0.014  n=968   lat=0.3s
  gemini       r=0.528±0.110  n=18    lat=0.72s
  ollama       r=0.402±0.127  n=12    lat=2.54s

Post-training policy (1000 samples):
  (he,torah): cerebras=98%, gemini=1%, ollama=0%
  (en,nt):    cerebras=82%, gemini=17%, ollama=1%
```

The router converged on the top arm for each context while still exploring the
runner-up when its reward was close.

## Production use

Validated in a high-throughput batch pipeline routing between Cerebras,
Ollama, and Gemini across two parallel workers (two languages, 31k tasks).
File-locked state is shared across processes; `record()` is atomic.

## State & persistence

- Location: `~/.rl_router_state.json` (override via `RL_ROUTER_STATE` env or
  `Router(state_path=...)`).
- Format: JSON, human-readable, ~1KB per context.
- Writes: atomic replace + fsync. Safe under kill.
- Reads: shared lock (POSIX: `fcntl.LOCK_SH`, Windows: `msvcrt.LK_NBLCK`).

## Config knobs (rarely needed)

```python
from rl_router import Router, RouterConfig

r = Router(
    arms=["a", "b"],
    config=RouterConfig(
        half_life_calls=500,        # how fast old data decays
        target_latency_s=2.0,       # latency scale for reward
        rate_limit_penalty=0.5,     # extra cost for 429
        exploration_floor=0.02,     # ε of pure random picks
        cooldown_s=60.0,            # 429'd arm penalty window
    ),
)
```

## Observability

```python
print(r.pretty())
# [en|summarization]
#   openai       r=0.891±0.012  n=1204  lat=0.42s
#   anthropic    r=0.854±0.019  n=387   lat=0.58s
#   local        r=0.312±0.089  n=22    lat=3.1s
```

Hook `r.stats()` into your dashboard for per-context arm reward + latency
rollup.

## What this library is NOT

- Not an HTTP client. You call providers; `rl-router` only picks one.
- Not a cost optimizer. Reward is latency+success. Wrap your own cost signal
  into `record()` if you want cost-aware routing.
- Not a replacement for circuit breakers. Combine with one — rl-router
  degrades gracefully but doesn't open/close circuits.
- Not a drop-in for LiteLLM / OpenRouter. Those are proxies. This is a
  decision function. Use either or both.

## License

MIT.

## Status

0.1.0 — API stable for `pick` / `record` / `stats`. Internal state format may
evolve; upgrade path preserved via `state['v']` field.
