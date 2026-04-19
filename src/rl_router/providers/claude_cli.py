"""Claude Code CLI provider adapter.

Invokes the ``claude`` binary (installed via ``npm i -g @anthropic-ai/claude-code``)
in non-interactive print mode (``claude -p "<prompt>" --max-turns 1``).

Why a subprocess and not the Anthropic HTTP API?
    The Claude Code CLI is billed through the user's flat-rate Claude.ai
    subscription, not per-token API usage. For a subscription holder the
    effective cost is $0 per call, so it is a strictly better arm than the
    pay-per-token API when it is available. This module deliberately does
    **not** fall back to the HTTP API on failure — that is the RL Router's
    job (other arms pick up the slack).

The provider is designed to plug into :class:`rl_router.Router`:

    >>> from rl_router import Router
    >>> from rl_router.providers import ClaudeCLIProvider
    >>> claude = ClaudeCLIProvider()
    >>> r = Router(arms=["claude_cli", "cerebras", "ollama"],
    ...            priors={"claude_cli": (10.0, 1.0)})   # strong positive prior
    >>> arm = r.pick(("en", "chat"))
    >>> if arm == "claude_cli":
    ...     res = claude.call("Summarise the Iliad in one sentence.")
    ...     r.record(("en", "chat"), arm,
    ...              success=res.success, latency_s=res.latency_s)
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ClaudeCLIResult:
    """Outcome of a single ``claude -p`` invocation.

    ``cost_usd`` is always 0.0: the subscription is flat-rate, so per-call
    cost accounting is not meaningful. Exposed so calling code can uniformly
    sum costs across heterogeneous providers.
    """

    success: bool
    output: str
    latency_s: float
    return_code: int
    stderr: str
    timed_out: bool
    cost_usd: float = 0.0


class ClaudeCLIProvider:
    """Thin wrapper over the Claude Code CLI.

    Parameters
    ----------
    binary:
        Path or name of the ``claude`` executable. Defaults to looking up
        ``claude`` on ``$PATH``. On Windows, ``claude.cmd`` is resolved
        automatically by :func:`shutil.which`.
    timeout_s:
        Hard wall-clock cap for a single call. Default 60s.
    extra_args:
        Extra CLI arguments appended between ``-p`` and the prompt.
        Defaults to ``("--max-turns", "1")`` which keeps non-interactive
        calls single-shot.
    """

    DEFAULT_EXTRA_ARGS: tuple[str, ...] = ("--max-turns", "1")

    def __init__(
        self,
        binary: Optional[str] = None,
        timeout_s: float = 60.0,
        extra_args: Optional[Sequence[str]] = None,
    ) -> None:
        resolved = binary or shutil.which("claude") or "claude"
        self.binary = resolved
        self.timeout_s = float(timeout_s)
        self.extra_args: tuple[str, ...] = (
            tuple(extra_args) if extra_args is not None else self.DEFAULT_EXTRA_ARGS
        )

    # Small indirection so tests can monkey-patch without importing subprocess.
    def _run(self, cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def call(self, prompt: str) -> ClaudeCLIResult:
        """Run the CLI once with ``prompt`` and return the outcome.

        Never raises for backend failures; exit-code / timeout / missing
        binary all surface as ``success=False`` results so the RL Router
        can penalise the arm without a try/except wrapper at the call site.
        """
        cmd: list[str] = [self.binary, "-p", *self.extra_args, prompt]
        t0 = time.monotonic()
        try:
            proc = self._run(cmd, self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ClaudeCLIResult(
                success=False,
                output="",
                latency_s=elapsed,
                return_code=-1,
                stderr=stderr or f"timeout after {self.timeout_s}s",
                timed_out=True,
            )
        except FileNotFoundError as exc:
            elapsed = time.monotonic() - t0
            return ClaudeCLIResult(
                success=False,
                output="",
                latency_s=elapsed,
                return_code=-1,
                stderr=f"claude CLI not found: {exc}",
                timed_out=False,
            )
        except OSError as exc:
            elapsed = time.monotonic() - t0
            return ClaudeCLIResult(
                success=False,
                output="",
                latency_s=elapsed,
                return_code=-1,
                stderr=f"OS error launching claude CLI: {exc}",
                timed_out=False,
            )

        elapsed = time.monotonic() - t0
        success = proc.returncode == 0
        return ClaudeCLIResult(
            success=success,
            output=proc.stdout or "",
            latency_s=elapsed,
            return_code=proc.returncode,
            stderr=proc.stderr or "",
            timed_out=False,
        )

    def is_available(self) -> bool:
        """True iff a ``claude`` binary can be located on PATH."""
        return shutil.which(self.binary) is not None or shutil.which("claude") is not None
