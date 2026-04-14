# Routing Claude Code between Opus and Qwen3.6-Plus (free, via OpenRouter)

> **Source of the tip:** community post by @RoundtableSpace on X (Apr 2026) — not an
> official Anthropic integration. Verify tier pricing and model availability on
> OpenRouter before committing a production workload; the `:free` tier can disappear.

## The context

Claude Code respects three environment variables that override its default
backend:

```bash
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="sk-or-v1-<your-openrouter-key>"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen/qwen3.6-plus-preview:free"
```

With those set, the `claude` CLI talks to OpenRouter using Anthropic wire format,
and OpenRouter proxies to whatever model you named — in this case a free Qwen
preview with a 1M-token window and native multimodal input.

You can get an OpenRouter key for free at
`https://openrouter.ai/settings/keys`.

## Where rl-router fits

`rl-router` isn't an HTTP proxy — it's a decision function. You still make your
own calls to providers. So the pattern for combining the tip above with this
library is:

1. Treat each **effective backend** as a named arm: `qwen-free` (via OpenRouter
   with the env vars above) and `claude-opus` (with your Anthropic key against
   the official API).
2. Route per context and feed back success/latency.

```python
import os, time
from rl_router import Router

router = Router(arms=["qwen-free", "claude-opus"])

# Minimal client adapters — replace with your real SDK calls.
def call_qwen_free(prompt: str) -> str:
    # Hit OpenRouter directly; don't rely on ANTHROPIC_* env leaking into here.
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api",
    )
    resp = client.messages.create(
        model="qwen/qwen3.6-plus-preview:free",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_claude_opus(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


PROVIDERS = {"qwen-free": call_qwen_free, "claude-opus": call_claude_opus}


def ask(context, prompt: str, max_retries: int = 2):
    tried: set[str] = set()
    for _ in range(max_retries):
        arm = router.pick(context)
        if arm in tried:
            remaining = [a for a in router.arms if a not in tried]
            if not remaining:
                break
            arm = remaining[0]
        tried.add(arm)
        t0 = time.time()
        try:
            out = PROVIDERS[arm](prompt)
        except Exception:
            router.record(context, arm, success=False, latency_s=time.time() - t0)
            continue
        router.record(context, arm, success=True, latency_s=time.time() - t0)
        return out, arm
    raise RuntimeError("all arms exhausted")


# Example: use different contexts so the router can learn separate policies.
out, arm = ask(("en", "code"),  "Explain the Zen of Python in 100 words.")
out, arm = ask(("en", "prose"), "Write a 3-paragraph product update email.")
```

## What the router will learn

With a hybrid reward of `success × 1/(1+latency/2s) − 0.5×rate_limited`, the
router typically converges on:

- `qwen-free` for contexts where Qwen is "good enough" and latency/cost are
  flat. OpenRouter's free tier is the dominant arm here because the cost side
  isn't part of the reward — if you want cost-aware routing, fold it into
  `record()` by wrapping latency with a cost multiplier.
- `claude-opus` for contexts where Qwen's answers are wrong or truncated often
  enough that the success term dominates.

If the free tier starts rate-limiting, the 60-second cooldown and Beta-decay
shift traffic to Opus automatically until it recovers.

## Caveats (read before shipping)

- The tip above relies on Claude Code accepting a non-Anthropic base URL via
  `ANTHROPIC_BASE_URL`. That behavior was documented for bring-your-own-gateway
  setups; it is not an Anthropic-endorsed path to Qwen. Anthropic can change
  the behavior at any time.
- OpenRouter's `:free` pricing tier is a moving target; verify at
  https://openrouter.ai/models?q=qwen before routing production load.
- Benchmarks like "trades blows with Opus on coding" are from the community
  thread, not independently reproduced here. Run your own evals on your own
  prompts before relying on them.
- Never hardcode keys. Load them from a secret store or environment file that
  isn't checked into git.

## License

Same as the rest of this repository (MIT). See `../LICENSE`.
