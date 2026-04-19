"""Integration test -- hits the real NVIDIA build.nvidia.com endpoint.

Gated by ``pytest -m integration`` AND by a real ``NVIDIA_API_KEY`` in env.
Skipped automatically when the key is absent so CI stays green without
credentials.

Usage (from rl-router repo root)::

    export NVIDIA_API_KEY="nvapi-..."
    pytest -m integration tests/integration/test_nvidia_real.py -v -s

The test hits ``/v1/chat/completions`` once with a 5-token ``ping`` prompt
against ``nvidia/llama-3.1-nemotron-70b-instruct`` (Nemotron is NVIDIA's
own instruct-tuned Llama-3.1-70B; free-tier friendly on build.nvidia.com).

Hard rule: NEVER print the full API key. Only ``key[:10] + "..."``.
"""

from __future__ import annotations

import os

import pytest

from rl_router.providers import NvidiaBuildProvider


INTEGRATION_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY"),
        reason="NVIDIA_API_KEY not set -- skipping live API integration test",
    ),
]


def test_real_ping_nonstreaming(capsys: pytest.CaptureFixture) -> None:
    """Send a 5-token 'ping' prompt and assert we get a real response."""
    key = os.environ["NVIDIA_API_KEY"]
    masked = key[:10] + "..."
    print(f"\n[integration] using NVIDIA_API_KEY={masked} model={INTEGRATION_MODEL}")

    provider = NvidiaBuildProvider(
        model=INTEGRATION_MODEL,
        timeout_s=45.0,
        max_retries=2,
        backoff_base_s=1.0,
    )
    assert provider.is_available(), "provider should be available once key is set"

    result = provider.call("Say the word pong.", max_tokens=16)

    # Emit diagnostics for humans running with -s even when assertions pass.
    preview = (result.output or "")[:200].replace("\n", " ")
    print(
        f"[integration] status={result.status_code} "
        f"latency={result.latency_s:.2f}s "
        f"prompt_tokens={result.prompt_tokens} "
        f"completion_tokens={result.completion_tokens} "
        f"total_tokens={result.total_tokens} "
        f"retries={result.retries} "
        f"finish_reason={result.finish_reason!r} "
        f"preview={preview!r}"
    )

    assert result.success, f"expected success, got error={result.error!r}"
    assert result.status_code == 200, f"expected HTTP 200, got {result.status_code}"
    assert result.output, "expected non-empty output"
    assert result.total_tokens > 0, "expected upstream usage.total_tokens > 0"
    assert result.latency_s < 45.0, "latency under timeout"


def test_real_ping_streaming(capsys: pytest.CaptureFixture) -> None:
    """Stream the same ping prompt via SSE; assert at least one chunk arrives."""
    key = os.environ["NVIDIA_API_KEY"]
    masked = key[:10] + "..."
    print(f"\n[integration] stream test NVIDIA_API_KEY={masked}")

    provider = NvidiaBuildProvider(
        model=INTEGRATION_MODEL,
        timeout_s=45.0,
        max_retries=2,
    )

    chunks_iter, finalize = provider.stream("Say the word pong.", max_tokens=16)
    chunks: list[str] = []
    for piece in chunks_iter:
        chunks.append(piece)
    result = finalize()

    preview = ("".join(chunks))[:200].replace("\n", " ")
    print(
        f"[integration stream] status={result.status_code} "
        f"latency={result.latency_s:.2f}s "
        f"chunk_count={len(chunks)} "
        f"prompt_tokens={result.prompt_tokens} "
        f"completion_tokens={result.completion_tokens} "
        f"total_tokens={result.total_tokens} "
        f"finish_reason={result.finish_reason!r} "
        f"preview={preview!r}"
    )

    assert result.streamed, "result should be flagged streamed=True"
    assert result.status_code == 200, f"stream status expected 200, got {result.status_code}"
    assert len(chunks) >= 1, "expected at least one streamed chunk"
    assert result.output, "expected non-empty assembled output"
