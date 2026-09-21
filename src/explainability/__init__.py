"""
==============================================================================
Explainable AI Module (Phase 9 Alias).
Re-exports symbols from src.xai for full backwards and alternate-import compatibility.
==============================================================================
"""

from src.xai import (
    EvidenceRecord,
    FeatureAttributor,
    EvidenceExtractor,
    GradCAMInterface,
    VisualExplainer,
    ExplanationFormatter,
    ExplanationEngine,
)

__all__ = [
    "EvidenceRecord",
    "FeatureAttributor",
    "EvidenceExtractor",
    "GradCAMInterface",
    "VisualExplainer",
    "ExplanationFormatter",
    "ExplanationEngine",
]
