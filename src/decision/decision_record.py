"""
==============================================================================
Decision Record and Risk State Definitions (Phase 7).
Defines surveillance risk states and the structured DecisionEvent record.
==============================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import time
from typing import Any, Dict, List, Optional


class RiskState(str, Enum):
    """
    Formal surveillance risk states for the intelligent decision engine.
    """
    NORMAL = "NORMAL"           # Nominal, baseline surveillance conditions
    SUSPICIOUS = "SUSPICIOUS"   # Atypical movements or acoustic events; increased scrutiny
    HIGH_RISK = "HIGH_RISK"     # Coordinated rapid dispersion, physical struggle, or elevated anomaly
    EMERGENCY = "EMERGENCY"     # Confirmed severe threat, weapon, violent altercation, or mass panic
    RECOVERY = "RECOVERY"       # Post-emergency stabilization window verifying threat cessation

    def __str__(self) -> str:
        return self.value


@dataclass
class DecisionEvent:
    """
    Structured decision event emitted by the Intelligent Decision Engine.
    Represents the verified surveillance risk state, confidence, causal reasons,
    and underlying multimodal evidence breakdown.
    """
    timestamp: float                                      # Epoch timestamp
    previous_state: str                                   # Prior risk state
    current_state: str                                    # Evaluated risk state
    score: float                                          # Fused multimodal anomaly score [0.0 - 1.0]
    confidence: float                                     # Decision certainty [0.0 - 1.0]
    reason: str                                           # Objective human-readable justification
    dominant_modality: str = "none"                       # "video", "audio", "bimodal", "none"
    video_score: float = 0.0                              # Composite visual score [0.0 - 1.0]
    audio_score: float = 0.0                              # Composite acoustic score [0.0 - 1.0]
    weapon_score: float = 0.0                             # Dangerous object score [0.0 - 1.0]
    fight_score: float = 0.0                              # Altercation score [0.0 - 1.0]
    corroborated: bool = False                            # True if multi-sensor synergy confirmed
    critical_evidence: bool = False                       # True if severe threat rule triggered
    confirmed: bool = True                                # True if temporal persistence quota met
    transition: bool = False                              # True if state changed in this cycle
    dynamic_thresholds: Dict[str, float] = field(default_factory=dict)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision record into a JSON-serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 4),
            "previous_state": str(self.previous_state),
            "current_state": str(self.current_state),
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "dominant_modality": self.dominant_modality,
            "video_score": round(self.video_score, 4),
            "audio_score": round(self.audio_score, 4),
            "weapon_score": round(self.weapon_score, 4),
            "fight_score": round(self.fight_score, 4),
            "corroborated": bool(self.corroborated),
            "critical_evidence": bool(self.critical_evidence),
            "confirmed": bool(self.confirmed),
            "transition": bool(self.transition),
            "dynamic_thresholds": {k: round(v, 4) for k, v in self.dynamic_thresholds.items()},
            "evidence_summary": self.evidence_summary,
        }

    def to_json(self) -> str:
        """Serialize decision record to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2)
