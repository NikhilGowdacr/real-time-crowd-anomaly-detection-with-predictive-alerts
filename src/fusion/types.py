"""
==============================================================================
Multimodal Fusion Subsystem Types and Data Structures (Phase 6).
Structured containers for cross-modal evidence, synergy detection,
and consolidated surveillance risk assessments.
==============================================================================
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModalityEvidence:
    """
    Standardized container representing evidence telemetry from a single sensory modality.
    """
    modality: str                             # "video" or "audio"
    score: float                              # Anomaly score [0.0 - 1.0]
    confidence: float                         # Confidence level [0.0 - 1.0]
    event_label: str                          # e.g., "scattering", "scream", "normal"
    timestamp: float                          # Epoch timestamp
    is_active: bool = True                    # False if sensor stream is offline/stale
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "event_label": self.event_label,
            "timestamp": self.timestamp,
            "is_active": self.is_active,
            "details": self.details,
        }


@dataclass
class CrossModalSynergy:
    """
    Represents detected cross-modal correlation between acoustic and visual events.
    Mutually confirming sensory patterns boost the composite risk score and certainty.
    """
    synergy_detected: bool = False
    synergy_type: Optional[str] = None         # e.g. "SCREAM_AND_SCATTER", "FIGHT_AND_SHOUTING"
    boost_multiplier: float = 1.0              # e.g. 1.25x for confirmed cross-modal emergencies
    description: str = "No cross-modal synergy detected."
    correlated_events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "synergy_detected": self.synergy_detected,
            "synergy_type": self.synergy_type,
            "boost_multiplier": round(self.boost_multiplier, 3),
            "description": self.description,
            "correlated_events": self.correlated_events,
        }


@dataclass(init=False)
class MultimodalAssessment:
    """
    Consolidated, synchronized multimodal anomaly and safety assessment.
    Serves as the unified risk record for surveillance HUD rendering,
    downstream Phase 7 Decision Engine, and Phase 11 Database logging.
    """
    multimodal_score: float = 0.0             # Final fused anomaly score [0.0 - 1.0]
    multimodal_confidence: float = 1.0        # Joint confidence level [0.0 - 1.0]
    video_score: float = 0.0                  # Constituent composite video score [0.0 - 1.0]
    audio_score: float = 0.0                  # Constituent audio anomaly score [0.0 - 1.0]
    status: str = "NORMAL"                    # "NORMAL", "SUSPICIOUS", "EMERGENCY"
    dominant_modality: str = "none"           # "video", "audio", "bimodal", or "none"
    synergy: CrossModalSynergy = field(default_factory=CrossModalSynergy)
    false_positive_suppressed: bool = False   # True if uncorroborated single-sensor noise was dampened
    suppression_reason: Optional[str] = None
    evidence_breakdown: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = "Nominal environment."

    def __init__(
        self,
        multimodal_score: float = 0.0,
        multimodal_confidence: float = 1.0,
        video_score: float = 0.0,
        audio_score: float = 0.0,
        status: str = "NORMAL",
        dominant_modality: str = "none",
        synergy: Optional[CrossModalSynergy] = None,
        false_positive_suppressed: bool = False,
        suppression_reason: Optional[str] = None,
        evidence_breakdown: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        description: str = "Nominal environment.",
        **kwargs: Any,
    ):
        if "anomaly_score" in kwargs:
            multimodal_score = float(kwargs.pop("anomaly_score"))
        if "confidence" in kwargs:
            multimodal_confidence = float(kwargs.pop("confidence"))

        self.multimodal_score = float(multimodal_score)
        self.multimodal_confidence = float(multimodal_confidence)
        self.video_score = float(video_score)
        self.audio_score = float(audio_score)
        self.status = str(status)
        self.dominant_modality = str(dominant_modality)
        self.synergy = synergy if synergy is not None else CrossModalSynergy()
        self.false_positive_suppressed = bool(false_positive_suppressed)
        self.suppression_reason = suppression_reason
        self.evidence_breakdown = evidence_breakdown if evidence_breakdown is not None else {}
        if kwargs:
            self.evidence_breakdown.update(kwargs)
        self.timestamp = float(timestamp) if timestamp is not None else time.time()
        self.description = str(description)

    @property
    def anomaly_score(self) -> float:
        return self.multimodal_score

    @property
    def confidence(self) -> float:
        return self.multimodal_confidence

    @property
    def corroborated(self) -> bool:
        return bool(self.synergy and self.synergy.synergy_detected)

    @property
    def weights(self) -> Dict[str, float]:
        return self.evidence_breakdown.get("weights", {"video": 0.60, "audio": 0.40})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON logging and SQLite database storage."""
        return {
            "timestamp": self.timestamp,
            "multimodal_score": round(self.multimodal_score, 4),
            "multimodal_confidence": round(self.multimodal_confidence, 4),
            "video_score": round(self.video_score, 4),
            "audio_score": round(self.audio_score, 4),
            "status": self.status,
            "dominant_modality": self.dominant_modality,
            "synergy_detected": self.synergy.synergy_detected,
            "synergy_type": self.synergy.synergy_type,
            "boost_multiplier": self.synergy.boost_multiplier,
            "false_positive_suppressed": self.false_positive_suppressed,
            "suppression_reason": self.suppression_reason,
            "evidence_breakdown": self.evidence_breakdown,
            "description": self.description,
        }
