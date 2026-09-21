"""
==============================================================================
Alert Record and Severity Definitions (Phase 8).
Defines alert severity levels, lifecycle status enums, and structured AlertRecord.
==============================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import time
from typing import Any, Dict, Optional


class AlertSeverity(str, Enum):
    """
    Operational alert severity levels for surveillance alerts.
    Mapped directly from Phase 7 RiskState values.
    """
    INFO = "INFO"          # Informational notice (e.g. system recovery stabilization)
    WARNING = "WARNING"    # Early agitation or acoustic anomaly under scrutiny (SUSPICIOUS)
    HIGH = "HIGH"          # Sustained crowd divergence or confirmed struggle (HIGH_RISK)
    CRITICAL = "CRITICAL"  # Severe confirmed crisis, weapon, or mass panic (EMERGENCY)

    def __str__(self) -> str:
        return self.value

    @property
    def rank(self) -> int:
        """Numeric rank for severity comparison."""
        mapping = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4,
        }
        return mapping[self]

    def __lt__(self, other: "AlertSeverity") -> bool:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: "AlertSeverity") -> bool:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: "AlertSeverity") -> bool:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: "AlertSeverity") -> bool:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank >= other.rank


class AlertStatus(str, Enum):
    """
    Lifecycle states for an individual AlertRecord.
    Lifecycle updates do NOT alter underlying Phase 7 risk state.
    """
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"

    def __str__(self) -> str:
        return self.value


@dataclass
class AlertRecord:
    """
    Structured alert record emitted by AlertManager.
    Wraps confirmed Phase 7 DecisionEvent telemetry with lifecycle metadata
    and channel distribution statuses.
    """
    alert_id: str                                         # Unique identifier (e.g. ALT-20260910-192031-001)
    timestamp: float                                      # Epoch timestamp of alert emission
    severity: AlertSeverity                               # Mapped alert severity
    risk_state: str                                       # Underlying Phase 7 RiskState
    score: float                                          # Multimodal anomaly score [0.0 - 1.0]
    confidence: float                                     # Decision certainty [0.0 - 1.0]
    reason: str                                           # Objective observable evidence rationale
    dominant_modality: str = "none"                       # "video", "audio", "bimodal", "none"
    video_score: float = 0.0                              # Visual score [0.0 - 1.0]
    audio_score: float = 0.0                              # Acoustic score [0.0 - 1.0]
    weapon_score: float = 0.0                             # Dangerous object score [0.0 - 1.0]
    fight_score: float = 0.0                              # Altercation score [0.0 - 1.0]
    corroborated: bool = False                            # True if audio-visual synergy confirmed
    critical_evidence: bool = False                       # True if severe threat rule triggered
    source_event: Dict[str, Any] = field(default_factory=dict)
    channel_status: Dict[str, str] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[float] = None
    formatted_message: str = ""

    def __post_init__(self) -> None:
        if not self.formatted_message:
            self.formatted_message = self.format_human_message()

    def format_human_message(self) -> str:
        """
        Formats a clean, human-readable surveillance alert dispatch banner.
        Adheres to academic surveillance ethics: objective physical observations,
        zero defamatory/legal allegations.
        """
        sev_str = self.severity.value
        icon = "[CRITICAL]" if sev_str == "CRITICAL" else ("[WARNING]" if sev_str in ("HIGH", "WARNING") else "[INFO]")
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        corrob_str = "YES" if self.corroborated else "NO"
        crit_str = "YES" if self.critical_evidence else "NO"

        lines = [
            "=" * 50,
            f"{icon} {sev_str} CROWD SURVEILLANCE ALERT [{self.alert_id}]",
            "=" * 50,
            f"Time:              {time_str}",
            f"Risk State:        {self.risk_state}",
            f"Anomaly Score:     {self.score:.2f}",
            f"Decision Conf:     {self.confidence:.2f}",
            f"Dominant Modality: {self.dominant_modality}",
            f"Video Score:       {self.video_score:.2f}",
            f"Audio Score:       {self.audio_score:.2f}",
            f"Weapon Score:      {self.weapon_score:.2f}",
            f"Fight Score:       {self.fight_score:.2f}",
            f"Corroborated:      {corrob_str}",
            f"Critical Evidence: {crit_str}",
            "-" * 50,
            "Reason:",
            f"  {self.reason}",
            "-" * 50,
            "Recommended Action:",
        ]

        if sev_str == "CRITICAL":
            lines.append("  Immediate on-site operator verification and crowd safety protocol initiation.")
        elif sev_str == "HIGH":
            lines.append("  Dispatch security personnel to monitor crowd movement and investigate agitation.")
        elif sev_str == "WARNING":
            lines.append("  Elevate surveillance scrutiny on the active camera/acoustic sector.")
        else:
            lines.append("  Surveillance conditions nominal. Maintain standard monitoring.")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert record to a JSON-serializable dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": round(self.timestamp, 4),
            "severity": self.severity.value,
            "risk_state": str(self.risk_state),
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
            "source_event": self.source_event,
            "channel_status": self.channel_status,
            "status": self.status.value,
            "acknowledged": bool(self.acknowledged),
            "acknowledged_at": round(self.acknowledged_at, 4) if self.acknowledged_at else None,
            "resolved": bool(self.resolved),
            "resolved_at": round(self.resolved_at, 4) if self.resolved_at else None,
            "formatted_message": self.formatted_message,
        }

    def to_json(self) -> str:
        """Serialize alert record to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2)
