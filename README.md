# rl-router

**Contextual Thompson Sampling router for multi-provider LLM APIs.**
Zero config, zero hyperparameters to tune, zero Python dependencies. A single
~240-line module you can read in one sitting. Adapts online to rate limits,
quota resets, and latency drift.

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

That's the whole API. The router learns which provider wins per context.

---

## 🔥 rl-router is a powerful tool

rl-router gives you adaptive LLM provider routing via contextual Thompson
Sampling bandits with zero config, zero dependencies, sub-millisecond decisions.

**What it can do:**
- Learn optimal provider per task context from live feedback
- Survive rate limits, quota resets, and latency drift automatically
- Scale across parallel workers without synchronizing onto the same provider

**With great power comes responsibility.** This tool can route production LLM
traffic across providers — when misconfigured, it can also cause cost spikes or
reliability incidents if arms/rewards are misconfigured. Read the
[Disclaimer](DISCLAIMER.md) before deploying to production. You are solely
responsible for your use.

**Recommended for:**
- Production LLM pipelines running on multiple providers (OpenAI + Anthropic + local, etc.)
- Batch inference jobs with parallel workers needing adaptive routing

**NOT recommended for:**
- Life-critical, safety-critical, or mission-critical systems
- Regulatory environments requiring formal validation (medical, aerospace, SIL)
- Any use that would violate third-party terms of service

## Why a small-surface router matters now

On **2026-03-24**, two releases of the `litellm` package on PyPI —
**v1.82.7** and **v1.82.8** — were published with a backdoor. The publish
token was exfiltrated via a compromised Trivy GitHub Action in LiteLLM's own
CI pipeline; PyPI quarantined the artifacts about three hours later. The
incident was assigned **CVE-2026-33634**.

References:
- <https://snyk.io/blog/poisoned-security-scanner-backdooring-litellm/>
- <https://docs.litellm.ai/blog/security-update-march-2026>

Nothing about this is a knock on the LiteLLM maintainers — the attacker never
touched LiteLLM source code. What the incident exposes is a *structural* risk:
any LLM proxy or router that ships a wide transitive dependency tree is
effectively routing your API keys through whatever happens to land in
`site-packages`. Poisoning a CI tool can be enough.

`rl-router` takes the opposite bet: be small enough that you can audit the
whole thing yourself in one sitting, and leave the HTTP layer to SDKs you
already trust.

## Threat model

- **Zero third-party Python dependencies.** `pyproject.toml` has
  `dependencies = []`. The import graph is the Python standard library only
  (`json`, `math`, `os`, `random`, `tempfile`, `time`, `fcntl`/`msvcrt`,
  `pathlib`).
- **Not an HTTP client.** `rl-router` never touches your API keys, never opens
  a socket, never parses provider responses. It returns a string
  (`"openai"`, `"anthropic"`, …) and you call the provider yourself with
  whatever SDK you already trust.
- **Single file of logic.** `src/rl_router/router.py` is ~240 lines. Read it
  before you install it. The supply-chain surface is whatever PyPI
  infrastructure ships the wheel, plus `hatchling` at build time, plus
  CPython stdlib — nothing else.
- **State is a plain JSON file** at `~/.rl_router_state.json`. Human-readable,
  grep-able, diff-able in CI.

This is not a formal audit claim. It's a size claim: small enough that you
can audit it yourself.

## Comparative footprint

| Tool | Role | Python deps | Config | Adapts online |
|---|---|---|---|---|
| **rl-router** | Decision function | 0 | Code API, no config file | Yes — Thompson Sampling + decay + cooldown |
| LiteLLM | Proxy / SDK | Many (transitive tree) | YAML `model_list` + proxy config | Static fallback lists |
| OpenRouter | SaaS gateway | Client SDK only | JSON request body | Provider-side, opaque |
| Bifrost (Go proxy) | Proxy | N/A (Go binary) | YAML | Weighted round-robin / failover |
| Router-R1 (arXiv 2506.09033) | Research: LLM-as-router | PyTorch + transformers | Prompt-time | Yes — but routes via a full LLM |

Router-R1 is the closest research cousin (also learned routing), but uses a
full LLM to make the routing decision. `rl-router` is a conjugate-Bayesian
update on Beta(α, β) distributions — microseconds per pick, no GPU, no model
weights.

## Common pain points this addresses

Taken from real issues filed against other routers:

- **Unknown-model fallback** ([BerriAI/litellm#15114](https://github.com/BerriAI/litellm/issues/15114),
  [#25080](https://github.com/BerriAI/litellm/issues/25080) — fallback behavior
  when a requested model isn't in `model_list`): `rl-router`'s arms are opaque
  string labels you define. There is no central registry to be out of sync with.
- **Provider diversification under parallel load** (multiple workers
  stampeding the same provider): Thompson Sampling is non-deterministic, so
  parallel workers spread naturally instead of collapsing onto the same arm.
- **Quota reset recovery** (static fallback never retries a "dead" provider):
  exponential decay with half-life of 500 calls re-explores cooled-down arms
  automatically.
- **Rate-limit cooldown** (hammering a 429'd provider): 60-second cooldown
  with soft penalty, not a hard circuit-break.
- **Image / multimodal payload passthrough**: out of scope — `rl-router` never
  touches your request body. You pass whatever payload you want to whichever
  arm is picked. This is a feature of being a decision function, not a proxy.
- **Provider routing inside higher-level frameworks**
  ([run-llama/LlamaIndexTS#2262](https://github.com/run-llama/LlamaIndexTS/issues/2262),
  [simonw/llm-openrouter#17](https://github.com/simonw/llm-openrouter/issues/17)):
  `rl-router` is a ~240-line Python module; you wrap it around any client
  call in two lines. Node / CLI bindings: roadmap.
- **Cache-control header passthrough** (Anthropic prompt caching through
  proxies): out of scope — proxy-layer concern, not a routing-layer concern.
- **Strict tool-call ID validation across providers**: out of scope, same
  reason — could be a companion adapter package.

The honest framing: `rl-router` is a **decision function**. Issues that are
really about HTTP payload translation belong in a thin adapter layer sitting
next to it, not inside it.

## Why not just try-catch a provider list?

Hand-written fallback logic fails silently in three common ways:

| Failure mode | Static fallback | rl-router |
|---|---|---|
| Provider A hits rate limit mid-job | Keeps hammering A, retries noisy | Shifts weight to B in real time, cools A down 60s |
| Quota resets at midnight UTC | Never recovers A until you redeploy | Decay (half-life 500 calls) re-explores A automatically |
| Latency degrades on B | No signal, users complain | Reward penalizes slow success, picks C |
| Workers run in parallel (saturation) | All hit same provider, worsen 429 | Thompson sampling diversifies independently |

Bandits aren't new. What's new is **wiring them to the LLM ops problem with
zero config** and surviving real production pain (file-locked state, atomic
writes, decay, cooldown).

## Install

```
pip install rl-router
```

No dependencies. Python 3.9+. Works on Linux, macOS, Windows.

## ⚠️ Disclaimer

rl-router is provided **AS IS**, without warranty of any kind. Use at
your own risk. You are solely responsible for your use and for compliance
with all applicable laws and third-party terms of service.

See [DISCLAIMER.md](DISCLAIMER.md) for full terms.

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
- Not formally audited. Small enough that you can audit it yourself.

## License

MIT.

## Status

0.1.0 — API stable for `pick` / `record` / `stats`. Internal state format may
evolve; upgrade path preserved via `state['v']` field.
