"""
==============================================================================
Explainable AI (XAI) & Evidence Visualization Subsystem (Phase 9).
Provides transparent, auditable evidence records, deterministic factor
attribution, visual motion vector overlays, and structured explanations.
==============================================================================
"""

from src.xai.evidence_record import EvidenceRecord
from src.xai.feature_attribution import FeatureAttributor
from src.xai.evidence_extractor import EvidenceExtractor
from src.xai.gradcam_interface import GradCAMInterface
from src.xai.visual_explanation import VisualExplainer
from src.xai.explanation_formatter import ExplanationFormatter
from src.xai.explanation_engine import ExplanationEngine

__all__ = [
    "EvidenceRecord",
    "FeatureAttributor",
    "EvidenceExtractor",
    "GradCAMInterface",
    "VisualExplainer",
    "ExplanationFormatter",
    "ExplanationEngine",
]
