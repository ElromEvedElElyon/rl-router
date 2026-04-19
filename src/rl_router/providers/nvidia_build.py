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
and is expected to be loaded from ``E:\\Dev\\.secrets\\.env`` -- never hardcode
it here and never commit it to git.

Upgrades in this revision (Wave-9-NVIDIA-04)
--------------------------------------------
1. ``stream=True`` SSE support via :meth:`stream` -- yields delta text chunks
   and still accumulates a final :class:`NvidiaBuildResult` with usage stats.
2. Token usage tracking: :attr:`NvidiaBuildResult.prompt_tokens`,
   :attr:`completion_tokens`, :attr:`total_tokens` populated from the
   upstream ``usage`` field when present.
3. Exponential backoff with jitter on HTTP 429 -- configurable
   ``max_retries`` (default 3) with base delay 1s, multiplier 2, jitter
   +/-25%. Exhaustion still surfaces ``success=False`` + ``rate_limited=True``.
4. Context-length clamp per model -- :data:`MODEL_CONTEXT_WINDOWS` maps known
   NIM model ids to their context cap. :meth:`call`/:meth:`stream` clamp
   ``max_tokens`` so ``rough_prompt_tokens + max_tokens <= window``.

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
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "meta/llama-3.1-405b-instruct"

# Context windows for commonly-used NIM-hosted models. Values are the
# *total* context (prompt + completion) advertised by NVIDIA / the model
# card. When a model is not in this table we fall back to
# ``DEFAULT_CONTEXT_WINDOW`` and the clamp becomes a very loose upper bound.
MODEL_CONTEXT_WINDOWS: dict = {
    "meta/llama-3.1-405b-instruct": 128_000,
    "meta/llama-3.1-70b-instruct": 128_000,
    "meta/llama-3.1-8b-instruct": 128_000,
    "nvidia/llama-3.1-nemotron-70b-instruct": 128_000,
    "nvidia/llama-3.1-nemotron-51b-instruct": 128_000,
    "mistralai/mixtral-8x22b-instruct-v0.1": 65_536,
    "mistralai/mixtral-8x7b-instruct-v0.1": 32_768,
    "mistralai/mistral-7b-instruct-v0.3": 32_768,
    "google/gemma-2-27b-it": 8_192,
    "google/gemma-2-9b-it": 8_192,
    "microsoft/phi-3-medium-128k-instruct": 128_000,
    "microsoft/phi-3-mini-128k-instruct": 128_000,
    "qwen/qwen2.5-7b-instruct": 32_768,
    "qwen/qwen2.5-coder-32b-instruct": 32_768,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 16_384,
}

DEFAULT_CONTEXT_WINDOW = 8_192


def _rough_token_count(text: str) -> int:
    """Cheap heuristic: ~4 chars per token for English/code. Conservative."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    model: str = ""
    finish_reason: str = ""
    streamed: bool = False


class NvidiaBuildProvider:
    """Thin wrapper over the NVIDIA build.nvidia.com chat-completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = NVIDIA_DEFAULT_MODEL,
        base_url: str = NVIDIA_DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_s = max(0.0, float(backoff_base_s))
        self._sleep = sleep_fn if sleep_fn is not None else time.sleep

    # ------------------------------------------------------------------
    # HTTP indirection (mocked by unit tests)
    # ------------------------------------------------------------------

    def _http_post(self, url: str, body: bytes, headers: dict) -> tuple:
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

    def _http_post_stream(self, url: str, body: bytes, headers: dict) -> tuple:
        """Stream variant -- returns (status, line_iter, error)."""
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read() if hasattr(exc, "read") else b""
            except Exception:
                payload = b""

            def _err_iter():
                if payload:
                    yield payload
                return

            return exc.code, _err_iter(), f"HTTPError {exc.code}: {exc.reason}"

        def _iter():
            try:
                for line in resp:
                    yield line.rstrip(b"\r\n")
            finally:
                resp.close()

        return resp.status, _iter(), ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True iff an API key is configured."""
        return bool(self.api_key)

    def context_window(self, model: Optional[str] = None) -> int:
        """Return the known context window for ``model`` (or ``self.model``)."""
        name = model or self.model
        return MODEL_CONTEXT_WINDOWS.get(name, DEFAULT_CONTEXT_WINDOW)

    def _clamp_max_tokens(self, prompt: str, max_tokens: int) -> int:
        """Clamp ``max_tokens`` so rough-prompt + max_tokens <= context window.

        Leaves a 128-token safety margin for chat-template overhead.
        Returns at least 1 so callers never see a zero budget.
        """
        window = self.context_window()
        rough = _rough_token_count(prompt)
        budget = window - rough - 128
        if budget < 1:
            return 1
        return max(1, min(int(max_tokens), budget))

    def _compute_backoff(self, attempt: int) -> float:
        """Exponential backoff with +/-25% jitter. ``attempt`` is 0-indexed."""
        base = self.backoff_base_s * (2**attempt)
        jitter = base * 0.25
        return max(0.0, base + random.uniform(-jitter, jitter))

    def call(self, prompt: str, max_tokens: int = 1024) -> NvidiaBuildResult:
        """Send a single user message (non-streaming) and return the outcome."""
        if not self.api_key:
            return NvidiaBuildResult(
                success=False,
                output="",
                latency_s=0.0,
                status_code=0,
                error="NVIDIA_API_KEY not set",
                model=self.model,
            )

        clamped = self._clamp_max_tokens(prompt, max_tokens)
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": clamped,
                "stream": False,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        retries = 0
        t0 = time.monotonic()

        while True:
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
                    retries=retries,
                    model=self.model,
                )
            except OSError as exc:
                return NvidiaBuildResult(
                    success=False,
                    output="",
                    latency_s=time.monotonic() - t0,
                    status_code=0,
                    error=f"network error: {exc}",
                    retries=retries,
                    model=self.model,
                )

            elapsed = time.monotonic() - t0

            if status == 429 and retries < self.max_retries:
                self._sleep(self._compute_backoff(retries))
                retries += 1
                continue

            if status == 429:
                return NvidiaBuildResult(
                    success=False,
                    output="",
                    latency_s=elapsed,
                    status_code=status,
                    error=err or "429 Too Many Requests",
                    rate_limited=True,
                    retries=retries,
                    model=self.model,
                )

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
                    retries=retries,
                    model=self.model,
                )

            if status < 200 or status >= 300:
                return NvidiaBuildResult(
                    success=False,
                    output="",
                    latency_s=elapsed,
                    status_code=status,
                    error=err or f"HTTP {status}",
                    retries=retries,
                    model=self.model,
                )

            try:
                choice = data["choices"][0]
                text = choice["message"]["content"]
                finish_reason = str(choice.get("finish_reason") or "")
            except (KeyError, IndexError, TypeError):
                return NvidiaBuildResult(
                    success=False,
                    output="",
                    latency_s=elapsed,
                    status_code=status,
                    error="malformed response: no choices[0].message.content",
                    retries=retries,
                    model=self.model,
                )

            usage = data.get("usage") if isinstance(data, dict) else None
            prompt_tokens = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0
            completion_tokens = (
                int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
            )
            total_tokens = int(usage.get("total_tokens", 0)) if isinstance(usage, dict) else 0

            return NvidiaBuildResult(
                success=True,
                output=text or "",
                latency_s=elapsed,
                status_code=status,
                error="",
                retries=retries,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=self.model,
                finish_reason=finish_reason,
            )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, prompt: str, max_tokens: int = 1024):
        """Open an SSE chat-completion stream.

        Returns ``(text_chunks, finalize)``:

        * ``text_chunks`` -- generator yielding each ``delta.content`` string
          as it arrives. Consume to exhaustion to drain the socket.
        * ``finalize()`` -- call *after* draining the generator to get the
          assembled :class:`NvidiaBuildResult` (populated with latency, usage
          if the server sent a ``[DONE]`` block with usage, finish_reason,
          etc.).
        """
        if not self.api_key:
            empty_result = NvidiaBuildResult(
                success=False,
                output="",
                latency_s=0.0,
                status_code=0,
                error="NVIDIA_API_KEY not set",
                model=self.model,
                streamed=True,
            )

            def _empty_iter():
                return
                yield  # pragma: no cover

            return _empty_iter(), (lambda: empty_result)

        clamped = self._clamp_max_tokens(prompt, max_tokens)
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": clamped,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        state = {
            "chunks": [],
            "status": 0,
            "error": "",
            "rate_limited": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": "",
            "t0": time.monotonic(),
            "elapsed": 0.0,
        }

        try:
            status, line_iter, err = self._http_post_stream(url, body, headers)
        except TimeoutError as exc:
            state["error"] = f"timeout: {exc}"

            def _empty_iter():
                return
                yield

            return _empty_iter(), (lambda: NvidiaBuildResult(
                success=False,
                output="",
                latency_s=time.monotonic() - state["t0"],
                status_code=0,
                error=state["error"],
                timed_out=True,
                model=self.model,
                streamed=True,
            ))

        state["status"] = status
        state["error"] = err
        if status == 429:
            state["rate_limited"] = True

        def _gen():
            if status < 200 or status >= 300:
                for _ in line_iter:
                    pass
                state["elapsed"] = time.monotonic() - state["t0"]
                return
            for raw_line in line_iter:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    chunk = delta.get("content") or ""
                    finish = choices[0].get("finish_reason")
                    if chunk:
                        state["chunks"].append(chunk)
                        yield chunk
                    if finish:
                        state["finish_reason"] = str(finish)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    state["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
                    state["completion_tokens"] = int(usage.get("completion_tokens", 0))
                    state["total_tokens"] = int(usage.get("total_tokens", 0))
            state["elapsed"] = time.monotonic() - state["t0"]

        def _finalize() -> NvidiaBuildResult:
            elapsed = state["elapsed"] or (time.monotonic() - state["t0"])
            st = state["status"]
            if st < 200 or st >= 300:
                return NvidiaBuildResult(
                    success=False,
                    output="".join(state["chunks"]),
                    latency_s=elapsed,
                    status_code=st,
                    error=state["error"] or f"HTTP {st}",
                    rate_limited=state["rate_limited"],
                    model=self.model,
                    streamed=True,
                )
            text = "".join(state["chunks"])
            return NvidiaBuildResult(
                success=bool(text) or state["finish_reason"] == "stop",
                output=text,
                latency_s=elapsed,
                status_code=st,
                error="",
                prompt_tokens=state["prompt_tokens"],
                completion_tokens=state["completion_tokens"],
                total_tokens=state["total_tokens"],
                finish_reason=state["finish_reason"],
                model=self.model,
                streamed=True,
            )

        return _gen(), _finalize
