"""
==============================================================================
Evidence Record and XAI Data Structures (Phase 9).
Defines the structured EvidenceRecord container capturing transparent,
auditable evidence, deterministic feature contributions, and human-readable
diagnostic explanations for surveillance anomaly events.
==============================================================================
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvidenceRecord:
    """
    Structured Explainable AI (XAI) evidence record.
    
    Provides transparent diagnostic rationale explaining WHY a surveillance anomaly
    score, risk state, or alert was produced. Distinguishes strictly between:
    - Anomaly Score: Severity/magnitude of the observed event [0.0 - 1.0]
    - Decision Confidence: Statistical certainty of the decision engine [0.0 - 1.0]
    - Evidence Contribution: Relative normalized attribution of contributing factors [0.0 - 1.0]
    """
    timestamp: float                                         # Epoch timestamp
    risk_state: str                                          # "NORMAL", "SUSPICIOUS", "HIGH_RISK", "EMERGENCY", "RECOVERY"
    anomaly_score: float                                     # Fused multimodal anomaly score [0.0 - 1.0]
    decision_confidence: float                                # Decision certainty [0.0 - 1.0]
    dominant_modality: str = "none"                          # "video", "audio", "bimodal", "none"
    
    # Observable metric magnitudes
    crowd_density: float = 0.0                               # Measured crowd density (per m^2 or score)
    movement_score: float = 0.0                              # Crowd movement / velocity score [0.0 - 1.0]
    behavior_score: float = 0.0                              # Behavioral pattern anomaly score [0.0 - 1.0]
    audio_score: float = 0.0                                 # Acoustic anomaly score [0.0 - 1.0]
    fight_score: float = 0.0                                 # Physical struggle score [0.0 - 1.0]
    weapon_score: float = 0.0                                # Dangerous object score [0.0 - 1.0]
    corroborated: bool = False                               # True if multi-sensor cross-modal synergy confirmed
    critical_evidence: bool = False                          # True if critical safety override active
    
    # Causal and ranked explanations
    top_evidence: List[str] = field(default_factory=list)    # Ranked list of dominant observable factors
    evidence_contributions: Dict[str, float] = field(default_factory=dict)  # Normalized relative contributions (sum to 1.0 or 0.0)
    ranked_contributions: List[Tuple[str, float]] = field(default_factory=list) # Ordered (factor, weight) tuples
    
    # Textual formulations
    explanation: str = ""                                    # Detailed diagnostic narrative
    short_explanation: str = ""                              # Single-line operator summary
    hud_explanation: List[str] = field(default_factory=list) # 3-5 concise bullet points for surveillance HUD
    alert_rationale: Optional[str] = None                    # "WHY THIS ALERT WAS GENERATED" (if linked to alert)
    
    # Academic integrity & transparent model indicators
    visual_evidence_available: bool = True                   # Physical visual evidence (kinematics/bounding boxes)
    model_explanation_available: bool = False                # False when Grad-CAM/attention is unconfigured
    model_explanation_reason: str = "No trained explainable model configured"
    audio_details: Dict[str, Any] = field(default_factory=dict)  # Honest acoustic details (spectral baseline vs classifier)
    details: Dict[str, Any] = field(default_factory=dict)    # Additional diagnostic telemetry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence record into a JSON-serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 4),
            "risk_state": str(self.risk_state),
            "anomaly_score": round(self.anomaly_score, 4),
            "decision_confidence": round(self.decision_confidence, 4),
            "dominant_modality": self.dominant_modality,
            "metrics": {
                "crowd_density": round(self.crowd_density, 3),
                "movement_score": round(self.movement_score, 3),
                "behavior_score": round(self.behavior_score, 3),
                "audio_score": round(self.audio_score, 3),
                "fight_score": round(self.fight_score, 3),
                "weapon_score": round(self.weapon_score, 3),
                "corroborated": bool(self.corroborated),
                "critical_evidence": bool(self.critical_evidence),
            },
            "top_evidence": self.top_evidence,
            "evidence_contributions": {k: round(v, 4) for k, v in self.evidence_contributions.items()},
            "ranked_contributions": [(k, round(v, 4)) for k, v in self.ranked_contributions],
            "explanation": self.explanation,
            "short_explanation": self.short_explanation,
            "hud_explanation": self.hud_explanation,
            "alert_rationale": self.alert_rationale,
            "transparency": {
                "visual_evidence_available": self.visual_evidence_available,
                "model_explanation_available": self.model_explanation_available,
                "model_explanation_reason": self.model_explanation_reason,
                "audio_details": self.audio_details,
            },
            "details": self.details,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize evidence record to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
