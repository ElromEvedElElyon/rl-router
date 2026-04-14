# Launch material — DO NOT POST WITHOUT YOUR REVIEW

Everything below is **draft only**. Nothing was posted.

---

## Hacker News (Show HN)

**Title:** Show HN: rl-router – Thompson sampling bandit for LLM provider routing (zero config)

**Body:**

I kept writing the same provider-fallback try/except for every LLM pipeline I
touched. Static fallback chains fail in predictable ways: they keep hammering a
rate-limited provider, they don't recover when quota resets, and parallel
workers collapse onto the same provider and worsen 429s.

rl-router is a contextual multi-armed bandit using Thompson sampling. You give
it arms (providers), it gives you a pick per context (tuple like `(language,
task_type)`). You feed back success/latency/rate_limit. It learns.

Design choices:
- Thompson, not UCB or ε-greedy — parallel workers naturally diversify because
  sampling is stochastic.
- Exponential decay (half-life 500 calls) so old evidence fades and the router
  re-explores when quotas reset.
- 60s cooldown on a 429'd arm (additive, not hard ban).
- Zero dependencies, Python 3.9+, works on Windows because file locking is
  portable.
- Atomic JSON state, safe under kill.

Validated in a batch job running two parallel workers over 31k tasks across
three providers. Convergence in synthetic benchmark reaches ~98% on the best
arm per context in ~1k steps.

Repo: https://github.com/ElromEvedElElyon/rl-router
Install: `pip install rl-router`

---

## Tweet

rl-router: drop-in Thompson sampling for LLM provider routing.

• 0 config
• 0 deps
• learns the best provider per context (per language, per task, whatever)
• survives rate limits + quota resets + parallel workers
• Python 3.9+, cross-platform

pip install rl-router
https://github.com/ElromEvedElElyon/rl-router

---

## r/MachineLearning (self-post)

**Title:** [P] rl-router — contextual Thompson sampling for LLM API routing

**Body:**

Small library for a specific production problem: routing between N LLM
providers adaptively. Thompson sampling per context, Beta priors, decay for
quota resets, cooldown for 429s, atomic persistent state.

Comparison to the things I looked at before writing it:
- ε-greedy: wastes ε on known-bad arms forever
- UCB: deterministic → parallel workers sync onto the same arm
- hand-weighted fallback lists: no adaptation to drift
- RL libraries (tianshou, etc.): 1000x too heavy for "pick one of 3 HTTP
  endpoints"

Repo has a 2000-step synthetic benchmark that reproduces convergence. Open to
feedback on reward shaping (currently `success/(1+lat/2) - 0.5*rate_limited`).

Not monetizing this. MIT license.

---

## Posting checklist (run after your review)

- [ ] Create empty GitHub repo `ElromEvedElElyon/rl-router`, push the package
- [ ] Verify GitHub Actions CI is green
- [ ] `pip install build twine && python -m build && twine upload dist/*`
  (needs PyPI token in `.pypirc` or env `TWINE_PASSWORD`)
- [ ] Tag `v0.1.0` on GitHub
- [ ] Post HN around 8-9am ET Tuesday (peak discovery window)
- [ ] Tweet immediately after HN posts
- [ ] Reddit r/MachineLearning: wait 2-3 hours after HN to avoid rate-limit
  cross-link flag

## What you may want to add before launching

- A real-world case study section (anonymized metrics). Right now the only
  benchmark is synthetic.
- Integration guide for LangChain / LiteLLM.
- A `AsyncRouter` variant (current one is sync; ok because it only does local
  file I/O, but async users will ask).
