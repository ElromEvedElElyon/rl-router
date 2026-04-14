# Changelog

## 0.1.0 (initial public release)

- Thompson Sampling contextual bandit for LLM provider routing.
- Zero-dependency: Python 3.9+ stdlib only.
- Cross-platform (Linux, macOS, Windows — portable file locking).
- Atomic state persistence with fsync.
- Exponential decay (500-call half-life) for quota reset resilience.
- 60s cooldown on rate-limited arms.
- 2% exploration floor for safety.
- `Router`, `RouterConfig` public API.
- `python -m rl_router.simulate` reproducible benchmark.
- 7 unit tests covering convergence, persistence, isolation, rate-limit.
- CI on 12 matrix combos (3 OS × 4 Python versions).
