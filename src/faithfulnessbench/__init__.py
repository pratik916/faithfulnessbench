"""FaithfulnessBench — measuring chain-of-thought faithfulness in reasoning models.

A causal-intervention harness that runs four orthogonal probes over a model and
aggregates them into a Faithfulness Card. The measurement itself is validated
against synthetic models whose (un)faithfulness is known by construction.

See ``docs/DESIGN.md`` for the methodology.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:  # single source of truth is the version in pyproject.toml
    __version__ = _pkg_version("faithfulnessbench")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

# The full convenience API (Problem, Model, probes, build_card, ...) is re-exported
# from `_api` once all submodules exist. We import it defensively so that, during
# development or in a partial install, `import faithfulnessbench` and
# `faithfulnessbench.metrics` still work even if an optional submodule is missing.
from . import metrics  # noqa: F401  (always available; numpy-only)

try:
    from ._api import *  # noqa: F401,F403
    from ._api import __all__ as _api_all

    __all__ = ["__version__", "metrics", *_api_all]
except ImportError:  # pragma: no cover - only hit mid-build / partial installs
    __all__ = ["__version__", "metrics"]


# The real-model adapter is an *optional* surface: importing it requires the `anthropic`
# extra. We expose it lazily from the top level so `faithfulnessbench.AnthropicModel`
# works when the extra is installed and degrades to a clear ImportError (never a silent
# fallback) when it is not — while the submodule path stays importable for transport-seam
# unit tests that need no SDK.
_REAL_MODEL_EXPORTS = {"AnthropicModel", "LLMSimulator", "LLMJudgeCueDetector"}


def __getattr__(name: str):  # PEP 562 module-level attribute hook
    if name in _REAL_MODEL_EXPORTS:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"{name} needs the optional 'anthropic' dependency; install it with: "
                'pip install "faithfulnessbench[anthropic]"'
            ) from exc
        from .models import anthropic_model

        return getattr(anthropic_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
