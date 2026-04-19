"""Provider adapters for the RL Router.

Providers are plain callables / objects that wrap a concrete LLM backend.
The Router itself is backend-agnostic (its arms are just opaque string
identifiers), so providers can be wired in freely by user code.
"""

from .claude_cli import ClaudeCLIProvider, ClaudeCLIResult
from .nvidia_build import NvidiaBuildProvider, NvidiaBuildResult

__all__ = [
    "ClaudeCLIProvider",
    "ClaudeCLIResult",
    "NvidiaBuildProvider",
    "NvidiaBuildResult",
]
