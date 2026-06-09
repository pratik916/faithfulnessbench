"""The real-model API is discoverable and guarded, and the free-text-CoT caveat is
documented before any real model is scored (fb-0nc.1).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import faithfulnessbench

ROOT = Path(__file__).resolve().parents[1]
_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None


def test_adapter_classes_importable_via_submodule_without_anthropic():
    # The injectable transport seam means these import fine even with no anthropic SDK.
    from faithfulnessbench.models.anthropic_model import (
        AnthropicModel,
        LLMJudgeCueDetector,
        LLMSimulator,
    )

    assert AnthropicModel and LLMJudgeCueDetector and LLMSimulator


@pytest.mark.skipif(_HAS_ANTHROPIC, reason="anthropic installed; guard not exercised")
def test_top_level_real_model_import_raises_clear_importerror_without_anthropic():
    with pytest.raises(ImportError):
        faithfulnessbench.AnthropicModel


def test_unknown_top_level_attribute_raises_attributeerror():
    with pytest.raises(AttributeError):
        faithfulnessbench.NoSuchThing


def test_free_text_cot_caveat_is_documented():
    from faithfulnessbench.models import anthropic_model

    doc = (anthropic_model.__doc__ or "").lower()
    assert "free-text" in doc and ("simulator" in doc or "judge" in doc)
    readme = (ROOT / "README.md").read_text().lower()
    assert "llm judge" in readme
