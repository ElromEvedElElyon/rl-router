"""NVIDIA build.nvidia.com provider adapter.

Talks to the OpenAI-compatible endpoint at ``https://integrate.api.nvidia.com/v1``
using the key in the ``NVIDIA_API_KEY`` environment variable. Designed to slot
into :class:`rl_router.Router` as a fallback arm when Claude CLI / Cerebras /
Gemini are exhausted.

Why this provider is fallback-only
----------------------------------
Per the project's routing rule, Claude CLI (billed through the flat-rate
Claude.ai subscription) remains the preferred arm whenever it is available.
This NVIDIA arm exists to keep the swarm responsive during Claude CLI token
outages, not to replace it. The ``NVIDIA_API_KEY`` is a 12-month issued key
and is expected to be loaded from ``E:\\Dev\\.secrets\\.env`` — never hardcode
it here and never commit it to git.

Rate-limit handling
-------------------
When the upstream returns HTTP 429 or a body containing an explicit
``rate_limit`` error code, :attr:`NvidiaBuildResult.rate_limited` is set and
the :class:`rl_router.Router` cooldown logic can penalise the arm for
``cooldown_s`` seconds before picking it again.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "meta/llama-3.1-405b-instruct"


@dataclass
class NvidiaBuildResult:
    """Outcome of a single ``/chat/completions`` call."""

    success: bool
    output: str
    latency_s: float
    status_code: int
    error: str
    rate_limited: bool = False
    timed_out: bool = False


class NvidiaBuildProvider:
    """Thin wrapper over the NVIDIA build.nvidia.com chat-completions API.

    Parameters
    ----------
    api_key:
        API key. Defaults to ``os.environ["NVIDIA_API_KEY"]``. When neither
        is set, :meth:`is_available` returns False and :meth:`call` returns a
        failed result — the Router arm is simply skipped.
    model:
        Model name in ``namespace/name`` form as used by NVIDIA. Default is
        ``meta/llama-3.1-405b-instruct`` (listed free on build.nvidia.com).
        Override for e.g. ``mistralai/mixtral-8x22b-instruct-v0.1``.
    base_url:
        OpenAI-compatible base URL. Default is the public NVIDIA endpoint.
    timeout_s:
        Hard wall-clock cap for a single call. Default 60s.

    The provider never raises for upstream failures; every failure mode
    surfaces as ``success=False`` so the Router can penalise the arm without
    a try/except at the call site.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = NVIDIA_DEFAULT_MODEL,
        base_url: str = NVIDIA_DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    # Small indirection so tests can monkey-patch without real HTTP.
    def _http_post(self, url: str, body: bytes, headers: dict) -> tuple[int, bytes, str]:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status, resp.read(), ""
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read() if hasattr(exc, "read") else b""
            except Exception:
                payload = b""
            return exc.code, payload, f"HTTPError {exc.code}: {exc.reason}"

    def is_available(self) -> bool:
        """True iff an API key is configured."""
        return bool(self.api_key)

    def call(self, prompt: str, max_tokens: int = 1024) -> NvidiaBuildResult:
        """Send a single user message and return the outcome."""
        if not self.api_key:
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=0.0,
                status_code=0,
                error="NVIDIA_API_KEY not set",
            )

        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        t0 = time.monotonic()
        try:
            status, raw, err = self._http_post(url, body, headers)
        except TimeoutError as exc:
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=time.monotonic() - t0,
                status_code=0,
                error=f"timeout: {exc}",
                timed_out=True,
            )
        except OSError as exc:
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=time.monotonic() - t0,
                status_code=0,
                error=f"network error: {exc}",
            )

        elapsed = time.monotonic() - t0

        # Explicit 429 — classic rate limit.
        if status == 429:
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=elapsed,
                status_code=status,
                error=err or "429 Too Many Requests",
                rate_limited=True,
            )

        # Attempt to parse body even on non-2xx — some providers embed quota
        # errors in a 400 with a specific error code.
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = {}

        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            err_obj = data["error"]
            code = str(err_obj.get("code") or err_obj.get("type") or "").lower()
            msg = str(err_obj.get("message") or err_obj)
            rate_limited = (
                "rate_limit" in code
                or "rate-limit" in code
                or "quota" in code
                or "rate limit" in msg.lower()
            )
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=elapsed,
                status_code=status,
                error=msg,
                rate_limited=rate_limited,
            )

        if status < 200 or status >= 300:
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=elapsed,
                status_code=status,
                error=err or f"HTTP {status}",
            )

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=elapsed,
                status_code=status,
                error="malformed response: no choices[0].message.content",
            )

        return NvidiaBuildResult(
            success=True,
            output=text or "",
            latency_s=elapsed,
            status_code=status,
            error="",
        )
