"""FaithfulnessBench — measuring chain-of-thought faithfulness in reasoning models.

A causal-intervention harness that runs four orthogonal probes over a model and
aggregates them into a Faithfulness Card. The measurement itself is validated
against synthetic models whose (un)faithfulness is known by construction.

See ``docs/DESIGN.md`` for the methodology.
"""
from __future__ import annotations

__version__ = "0.1.0"

# The full convenience API (Problem, Model, probes, build_card, ...) is re-exported
# from `_api` once all submodules exist. We import it defensively so that, during
# development or in a partial install, `import faithfulnessbench` and
# `faithfulnessbench.metrics` still work even if an optional submodule is missing.
from . import metrics  # noqa: F401  (always available; numpy-only)

try:
    from ._api import *  # noqa: F401,F403
    from ._api import __all__ as _api_all

    __all__ = ["__version__", "metrics", *_api_all]
except Exception:  # pragma: no cover - only hit mid-build / partial installs
    __all__ = ["__version__", "metrics"]
