"""
Database Record Models - Phase 11.

Defines immutable/structured dataclass containers for database entities:
sessions, decision events, alerts, XAI evidence, and system metrics.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from src.database.serialization import (
    deserialize_bool,
    from_iso8601,
    from_json_str,
    serialize_bool,
    serialize_enum,
    to_iso8601,
    to_json_str,
)


@dataclass
class SessionRecord:
    """Represents a monitoring session from startup to shutdown."""
    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: Optional[str] = None
    source_type: Optional[str] = "synthetic"
    video_source: Optional[str] = None
    audio_enabled: bool = True
    status: str = "RUNNING"
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "source_type": self.source_type,
            "video_source": self.video_source,
            "audio_enabled": self.audio_enabled,
            "status": self.status,
            "metadata_json": dict(self.metadata_json),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        meta = data.get("metadata_json")
        if isinstance(meta, str):
            meta = from_json_str(meta)
        elif not isinstance(meta, dict):
            meta = {}
        return cls(
            id=data.get("id"),
            session_id=str(data["session_id"]),
            started_at=str(data["started_at"]),
            ended_at=data.get("ended_at"),
            source_type=data.get("source_type"),
            video_source=data.get("video_source"),
            audio_enabled=bool(data.get("audio_enabled", True)),
            status=str(data.get("status", "RUNNING")),
            metadata_json=meta,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "SessionRecord":
        return cls.from_dict(json.loads(json_str))


@dataclass
class EventRecord:
    """Historical record of an anomaly/decision event."""
    event_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    risk_state: str = "NORMAL"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: Optional[str] = None
    anomaly_score: Optional[float] = None
    decision_confidence: Optional[float] = None
    alert_severity: Optional[str] = None
    dominant_modality: Optional[str] = None
    people_count: Optional[int] = None
    crowd_density: Optional[float] = None
    average_speed: Optional[float] = None
    dispersion: Optional[float] = None
    directional_divergence: Optional[float] = None
    abnormal_track_count: Optional[int] = None
    video_score: Optional[float] = None
    audio_score: Optional[float] = None
    fusion_score: Optional[float] = None
    corroborated: Optional[bool] = None
    critical_evidence: Optional[bool] = None
    confirmed: Optional[bool] = None
    reason: Optional[str] = None
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "risk_state": self.risk_state,
            "anomaly_score": self.anomaly_score,
            "decision_confidence": self.decision_confidence,
            "alert_severity": self.alert_severity,
            "dominant_modality": self.dominant_modality,
            "people_count": self.people_count,
            "crowd_density": self.crowd_density,
            "average_speed": self.average_speed,
            "dispersion": self.dispersion,
            "directional_divergence": self.directional_divergence,
            "abnormal_track_count": self.abnormal_track_count,
            "video_score": self.video_score,
            "audio_score": self.audio_score,
            "fusion_score": self.fusion_score,
            "corroborated": self.corroborated,
            "critical_evidence": self.critical_evidence,
            "confirmed": self.confirmed,
            "reason": self.reason,
            "metadata_json": dict(self.metadata_json),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventRecord":
        meta = data.get("metadata_json")
        if isinstance(meta, str):
            meta = from_json_str(meta)
        elif not isinstance(meta, dict):
            meta = {}
        return cls(
            id=data.get("id"),
            event_id=str(data["event_id"]),
            session_id=data.get("session_id"),
            timestamp=str(data["timestamp"]),
            risk_state=str(data.get("risk_state", "NORMAL")),
            anomaly_score=float(data["anomaly_score"]) if data.get("anomaly_score") is not None else None,
            decision_confidence=float(data["decision_confidence"]) if data.get("decision_confidence") is not None else None,
            alert_severity=data.get("alert_severity"),
            dominant_modality=data.get("dominant_modality"),
            people_count=int(data["people_count"]) if data.get("people_count") is not None else None,
            crowd_density=float(data["crowd_density"]) if data.get("crowd_density") is not None else None,
            average_speed=float(data["average_speed"]) if data.get("average_speed") is not None else None,
            dispersion=float(data["dispersion"]) if data.get("dispersion") is not None else None,
            directional_divergence=float(data["directional_divergence"]) if data.get("directional_divergence") is not None else None,
            abnormal_track_count=int(data["abnormal_track_count"]) if data.get("abnormal_track_count") is not None else None,
            video_score=float(data["video_score"]) if data.get("video_score") is not None else None,
            audio_score=float(data["audio_score"]) if data.get("audio_score") is not None else None,
            fusion_score=float(data["fusion_score"]) if data.get("fusion_score") is not None else None,
            corroborated=bool(data["corroborated"]) if data.get("corroborated") is not None else None,
            critical_evidence=bool(data["critical_evidence"]) if data.get("critical_evidence") is not None else None,
            confirmed=bool(data["confirmed"]) if data.get("confirmed") is not None else None,
            reason=data.get("reason"),
            metadata_json=meta,
            created_at=str(data.get("created_at") or to_iso8601(time.time())),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "EventRecord":
        return cls.from_dict(json.loads(json_str))


@dataclass
class AlertRecord:
    """Persistent representation of a safety alert with operator lifecycle."""
    alert_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    severity: str = "WARNING"
    risk_state: str = "SUSPICIOUS"
    status: str = "ACTIVE"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: Optional[str] = None
    session_id: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    dominant_modality: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    fingerprint: Optional[str] = None
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "risk_state": self.risk_state,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
            "dominant_modality": self.dominant_modality,
            "status": self.status,
            "created_at": self.created_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "fingerprint": self.fingerprint,
            "metadata_json": dict(self.metadata_json),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertRecord":
        meta = data.get("metadata_json")
        if isinstance(meta, str):
            meta = from_json_str(meta)
        elif not isinstance(meta, dict):
            meta = {}
        return cls(
            id=data.get("id"),
            alert_id=str(data["alert_id"]),
            event_id=data.get("event_id"),
            session_id=data.get("session_id"),
            timestamp=str(data["timestamp"]),
            severity=str(data.get("severity", "WARNING")),
            risk_state=str(data.get("risk_state", "SUSPICIOUS")),
            score=float(data["score"]) if data.get("score") is not None else None,
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            reason=data.get("reason"),
            dominant_modality=data.get("dominant_modality"),
            status=str(data.get("status", "CREATED")),
            created_at=str(data.get("created_at") or to_iso8601(time.time())),
            acknowledged_at=data.get("acknowledged_at"),
            resolved_at=data.get("resolved_at"),
            fingerprint=data.get("fingerprint"),
            metadata_json=meta,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "AlertRecord":
        return cls.from_dict(json.loads(json_str))


# Model alias to avoid namespace conflict when importing both AlertRecord types
DBAlertRecord = AlertRecord


@dataclass
class EvidenceRecord:
    """Persistent representation of an XAI factor contribution."""
    evidence_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    factor: str = "unknown"
    contribution: float = 0.0
    event_id: Optional[str] = None
    alert_id: Optional[str] = None
    rank: Optional[int] = None
    statement: Optional[str] = None
    visual_available: bool = True
    model_explanation_available: bool = False
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "evidence_id": self.evidence_id,
            "event_id": self.event_id,
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "factor": self.factor,
            "contribution": self.contribution,
            "rank": self.rank,
            "statement": self.statement,
            "visual_available": self.visual_available,
            "model_explanation_available": self.model_explanation_available,
            "metadata_json": dict(self.metadata_json),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRecord":
        meta = data.get("metadata_json")
        if isinstance(meta, str):
            meta = from_json_str(meta)
        elif not isinstance(meta, dict):
            meta = {}
        return cls(
            id=data.get("id"),
            evidence_id=str(data["evidence_id"]),
            event_id=data.get("event_id"),
            alert_id=data.get("alert_id"),
            timestamp=str(data["timestamp"]),
            factor=str(data["factor"]),
            contribution=float(data["contribution"]),
            rank=int(data["rank"]) if data.get("rank") is not None else None,
            statement=data.get("statement"),
            visual_available=bool(data.get("visual_available", True)),
            model_explanation_available=bool(data.get("model_explanation_available", False)),
            metadata_json=meta,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "EvidenceRecord":
        return cls.from_dict(json.loads(json_str))


# Model alias to avoid namespace conflict when importing both EvidenceRecord types
DBEvidenceRecord = EvidenceRecord


@dataclass
class SystemMetricRecord:
    """Historical operational and system performance sample."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: Optional[str] = None
    fps: Optional[float] = None
    frame_latency_ms: Optional[float] = None
    dashboard_latency_ms: Optional[float] = None
    people_count: Optional[int] = None
    crowd_density: Optional[float] = None
    anomaly_score: Optional[float] = None
    audio_score: Optional[float] = None
    fusion_score: Optional[float] = None
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "fps": self.fps,
            "frame_latency_ms": self.frame_latency_ms,
            "dashboard_latency_ms": self.dashboard_latency_ms,
            "people_count": self.people_count,
            "crowd_density": self.crowd_density,
            "anomaly_score": self.anomaly_score,
            "audio_score": self.audio_score,
            "fusion_score": self.fusion_score,
            "metadata_json": dict(self.metadata_json),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemMetricRecord":
        meta = data.get("metadata_json")
        if isinstance(meta, str):
            meta = from_json_str(meta)
        elif not isinstance(meta, dict):
            meta = {}
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id"),
            timestamp=str(data["timestamp"]),
            fps=float(data["fps"]) if data.get("fps") is not None else None,
            frame_latency_ms=float(data["frame_latency_ms"]) if data.get("frame_latency_ms") is not None else None,
            dashboard_latency_ms=float(data["dashboard_latency_ms"]) if data.get("dashboard_latency_ms") is not None else None,
            people_count=int(data["people_count"]) if data.get("people_count") is not None else None,
            crowd_density=float(data["crowd_density"]) if data.get("crowd_density") is not None else None,
            anomaly_score=float(data["anomaly_score"]) if data.get("anomaly_score") is not None else None,
            audio_score=float(data["audio_score"]) if data.get("audio_score") is not None else None,
            fusion_score=float(data["fusion_score"]) if data.get("fusion_score") is not None else None,
            metadata_json=meta,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "SystemMetricRecord":
        return cls.from_dict(json.loads(json_str))
