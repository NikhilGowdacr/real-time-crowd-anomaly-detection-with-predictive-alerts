"""
Multimodal Decision Fusion Subsystem (Phase 6).
Fuses video features (crowd density, kinematics, behavioral ViT/Swin,
fight detection, weapon evidence) and audio features (acoustic anomaly,
sound classification, confidence, temporal confirmation) into a unified risk representation.
"""

from src.fusion.types import (
    ModalityEvidence,
    CrossModalSynergy,
    MultimodalAssessment,
)
from src.fusion.temporal_aligner import TemporalStreamAligner
from src.fusion.multimodal_fusion import MultimodalFusionEngine

__all__ = [
    "ModalityEvidence",
    "CrossModalSynergy",
    "MultimodalAssessment",
    "TemporalStreamAligner",
    "MultimodalFusionEngine",
]
