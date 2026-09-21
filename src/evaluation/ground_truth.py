"""
Ground-Truth Representation and Schemas - Phase 12.

Defines schemas and containers for ground-truth annotations, event categories,
and evaluation provenance. Enforces academic integrity: never fabricates labels.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, List, Optional


class EvaluationSource(str, Enum):
    """Explicit provenance of evaluation data."""
    REAL_DATASET = "REAL_DATASET"
    GROUND_TRUTH = "GROUND_TRUTH"
    SYNTHETIC_SCENARIO = "SYNTHETIC_SCENARIO"
    CONTROLLED_SYNTHETIC = "CONTROLLED_SYNTHETIC"
    CONTROLLED_TEST = "CONTROLLED_TEST"
    PERFORMANCE_BENCHMARK = "PERFORMANCE_BENCHMARK"
    UNAVAILABLE = "UNAVAILABLE"


class GroundTruthState(str, Enum):
    """Standardized discrete operational risk states."""
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    EMERGENCY = "EMERGENCY"
    RECOVERY = "RECOVERY"


class EventCategory(str, Enum):
    """Standardized ground-truth event category taxonomies."""
    NORMAL = "normal"
    NORMAL_FLOW = "normal_flow"
    OVERCROWDING = "overcrowding"
    RAPID_MOVEMENT = "rapid_movement"
    FIGHT = "fight"
    WEAPON = "weapon"
    PANIC = "panic"
    STAMPEDE = "stampede"
    AUDIO_DISTRESS = "audio_distress"
    MULTIMODAL_CRISIS = "multimodal_crisis"
    UNKNOWN = "unknown"


@dataclass(init=False)
class GroundTruthAnnotation:
    """
    Ground-truth annotation container for a single frame, window, or segment.
    All ground-truth fields are strictly optional to reflect unannotated surveillance data.
    """
    sample_id: str
    state: Optional[str] = None
    is_anomaly: Optional[int] = None  # 0 = normal, 1 = anomalous
    event_category: Optional[str] = None
    confidence: float = 1.0
    source: str = EvaluationSource.UNAVAILABLE.value
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        sample_id: str,
        state: Optional[Any] = None,
        is_anomaly: Optional[int] = None,
        event_category: Optional[Any] = None,
        confidence: float = 1.0,
        source: Any = EvaluationSource.UNAVAILABLE.value,
        timestamp: Optional[Any] = None,
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[Any] = None,
    ) -> None:
        self.sample_id = sample_id
        self.state = state.value if hasattr(state, "value") else (str(state) if state is not None else None)
        self.is_anomaly = is_anomaly
        cat = category if category is not None else event_category
        self.event_category = cat.value if hasattr(cat, "value") else (str(cat) if cat is not None else None)
        self.confidence = float(confidence)
        self.source = source.value if hasattr(source, "value") else str(source)
        self.timestamp = str(timestamp) if timestamp is not None else datetime.now(timezone.utc).isoformat()
        self.notes = notes
        self.metadata = dict(metadata or {})

    @property
    def category(self) -> Optional[str]:
        return self.event_category

    @property
    def has_valid_ground_truth(self) -> bool:
        """Returns True only when explicit state or binary anomaly label exists."""
        return (self.state is not None) or (self.is_anomaly is not None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "state": self.state,
            "is_anomaly": self.is_anomaly,
            "event_category": self.event_category,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthAnnotation":
        return cls(
            sample_id=str(data["sample_id"]),
            state=data.get("state"),
            is_anomaly=int(data["is_anomaly"]) if data.get("is_anomaly") is not None else None,
            event_category=data.get("event_category"),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", EvaluationSource.UNAVAILABLE.value)),
            timestamp=str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "GroundTruthAnnotation":
        return cls.from_dict(json.loads(json_str))
