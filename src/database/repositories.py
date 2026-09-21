"""
Repository Data Access Layer - Phase 11.

Provides parameterized CRUD operations and historical query analytics
for Sessions, Events, Alerts, Evidence, and System Metrics.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

from src.database.db_manager import DatabaseManager
from src.database.models import (
    AlertRecord,
    EventRecord,
    EvidenceRecord,
    SessionRecord,
    SystemMetricRecord,
)
from src.database.serialization import (
    deserialize_bool,
    serialize_bool,
    to_iso8601,
    to_json_str,
)

logger = logging.getLogger("crowd_anomaly.database.repositories")


class SessionRepository:
    """Data access for monitoring sessions."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def create_session(self, session: SessionRecord) -> SessionRecord:
        """Insert a new surveillance session record."""
        sql = """
        INSERT INTO sessions (
            session_id, started_at, ended_at, source_type,
            video_source, audio_enabled, status, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            session.session_id,
            to_iso8601(session.started_at),
            to_iso8601(session.ended_at),
            session.source_type,
            session.video_source,
            serialize_bool(session.audio_enabled),
            session.status,
            to_json_str(session.metadata_json),
        )
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            session.id = cursor.lastrowid
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Fetch session by session_id."""
        sql = "SELECT * FROM sessions WHERE session_id = ?;"
        rows = self.db.execute_query(sql, (session_id,))
        return SessionRecord.from_dict(rows[0]) if rows else None

    def end_session(
        self,
        session_id: str,
        status: str = "COMPLETED",
        ended_at: Optional[str] = None,
    ) -> bool:
        """Mark session as ended with timestamp and status."""
        end_time = to_iso8601(ended_at or datetime.now(timezone.utc))
        sql = "UPDATE sessions SET status = ?, ended_at = ? WHERE session_id = ?;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (status, end_time, session_id))
            return cursor.rowcount > 0

    def list_sessions(self, limit: int = 50) -> List[SessionRecord]:
        """Retrieve recent sessions ordered by start time descending."""
        sql = "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?;"
        rows = self.db.execute_query(sql, (limit,))
        return [SessionRecord.from_dict(r) for r in rows]


class EventRepository:
    """Data access for decision and anomaly events."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def insert_event(self, event: EventRecord) -> EventRecord:
        """
        Insert decision event record.
        Uses INSERT OR IGNORE for idempotent duplicate safety.
        """
        sql = """
        INSERT OR IGNORE INTO events (
            event_id, session_id, timestamp, risk_state, anomaly_score,
            decision_confidence, alert_severity, dominant_modality,
            people_count, crowd_density, average_speed, dispersion,
            directional_divergence, abnormal_track_count, video_score,
            audio_score, fusion_score, corroborated, critical_evidence,
            confirmed, reason, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            event.event_id,
            event.session_id,
            to_iso8601(event.timestamp),
            event.risk_state,
            event.anomaly_score,
            event.decision_confidence,
            event.alert_severity,
            event.dominant_modality,
            event.people_count,
            event.crowd_density,
            event.average_speed,
            event.dispersion,
            event.directional_divergence,
            event.abnormal_track_count,
            event.video_score,
            event.audio_score,
            event.fusion_score,
            serialize_bool(event.corroborated),
            serialize_bool(event.critical_evidence),
            serialize_bool(event.confirmed),
            event.reason,
            to_json_str(event.metadata_json),
            to_iso8601(event.created_at or datetime.now(timezone.utc)),
        )
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if cursor.lastrowid:
                event.id = cursor.lastrowid
        return event

    def get_event(self, event_id: str) -> Optional[EventRecord]:
        """Fetch event by unique event_id."""
        sql = "SELECT * FROM events WHERE event_id = ?;"
        rows = self.db.execute_query(sql, (event_id,))
        return EventRecord.from_dict(rows[0]) if rows else None

    def get_recent_events(self, limit: int = 50) -> List[EventRecord]:
        """Retrieve recent events ordered by timestamp descending."""
        sql = "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?;"
        rows = self.db.execute_query(sql, (limit,))
        return [EventRecord.from_dict(r) for r in rows]

    def query_events_by_state(self, risk_state: str, limit: int = 50) -> List[EventRecord]:
        """Query events matching specific risk state (e.g. EMERGENCY)."""
        sql = "SELECT * FROM events WHERE risk_state = ? ORDER BY timestamp DESC LIMIT ?;"
        rows = self.db.execute_query(sql, (risk_state, limit))
        return [EventRecord.from_dict(r) for r in rows]

    def query_events_by_severity(self, alert_severity: str, limit: int = 50) -> List[EventRecord]:
        """Query events matching specific alert severity (e.g. CRITICAL)."""
        sql = "SELECT * FROM events WHERE alert_severity = ? ORDER BY timestamp DESC LIMIT ?;"
        rows = self.db.execute_query(sql, (alert_severity, limit))
        return [EventRecord.from_dict(r) for r in rows]

    def query_events_by_time_range(
        self, start_time: str, end_time: str, limit: int = 100
    ) -> List[EventRecord]:
        """Query events occurring between start_time and end_time."""
        sql = """
        SELECT * FROM events
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC LIMIT ?;
        """
        rows = self.db.execute_query(sql, (to_iso8601(start_time), to_iso8601(end_time), limit))
        return [EventRecord.from_dict(r) for r in rows]

    def get_event_count(self) -> int:
        """Count total stored events."""
        rows = self.db.execute_query("SELECT count(*) as cnt FROM events;")
        return int(rows[0]["cnt"]) if rows else 0


class AlertRepository:
    """Data access for surveillance alerts and lifecycle updates."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def insert_alert(self, alert: AlertRecord) -> AlertRecord:
        """
        Insert alert record.
        Uses INSERT OR IGNORE for idempotent duplicate safety.
        """
        sql = """
        INSERT OR IGNORE INTO alerts (
            alert_id, event_id, session_id, timestamp, severity,
            risk_state, score, confidence, reason, dominant_modality,
            status, created_at, acknowledged_at, resolved_at,
            fingerprint, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            alert.alert_id,
            alert.event_id,
            alert.session_id,
            to_iso8601(alert.timestamp),
            alert.severity,
            alert.risk_state,
            alert.score,
            alert.confidence,
            alert.reason,
            alert.dominant_modality,
            alert.status,
            to_iso8601(alert.created_at or datetime.now(timezone.utc)),
            to_iso8601(alert.acknowledged_at),
            to_iso8601(alert.resolved_at),
            alert.fingerprint,
            to_json_str(alert.metadata_json),
        )
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if cursor.lastrowid:
                alert.id = cursor.lastrowid
        return alert

    def get_alert(self, alert_id: str) -> Optional[AlertRecord]:
        """Fetch alert by alert_id."""
        sql = "SELECT * FROM alerts WHERE alert_id = ?;"
        rows = self.db.execute_query(sql, (alert_id,))
        return AlertRecord.from_dict(rows[0]) if rows else None

    def get_alerts_for_event(self, event_id: str) -> List[AlertRecord]:
        """Fetch all alerts triggered by a specific event_id."""
        sql = "SELECT * FROM alerts WHERE event_id = ? ORDER BY timestamp DESC;"
        rows = self.db.execute_query(sql, (event_id,))
        return [AlertRecord.from_dict(r) for r in rows]

    def get_alerts_for_session(self, session_id: str, limit: int = 50) -> List[AlertRecord]:
        """Fetch alerts associated with a session_id."""
        sql = "SELECT * FROM alerts WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?;"
        rows = self.db.execute_query(sql, (session_id, limit))
        return [AlertRecord.from_dict(r) for r in rows]

    def update_alert_status(
        self,
        alert_id: str,
        status: str,
        acknowledged_at: Optional[str] = None,
        resolved_at: Optional[str] = None,
    ) -> bool:
        """
        Update alert status and operator lifecycle timestamps.
        """
        updates = ["status = ?"]
        params: List[Any] = [status]

        if acknowledged_at is not None:
            updates.append("acknowledged_at = ?")
            params.append(to_iso8601(acknowledged_at))

        if resolved_at is not None:
            updates.append("resolved_at = ?")
            params.append(to_iso8601(resolved_at))

        params.append(alert_id)
        sql = f"UPDATE alerts SET {', '.join(updates)} WHERE alert_id = ?;"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            return cursor.rowcount > 0

    def get_active_alerts(self) -> List[AlertRecord]:
        """Retrieve alerts with status ACTIVE or CREATED."""
        sql = "SELECT * FROM alerts WHERE status IN ('ACTIVE', 'CREATED') ORDER BY timestamp DESC;"
        rows = self.db.execute_query(sql)
        return [AlertRecord.from_dict(r) for r in rows]

    def get_recent_alerts(self, limit: int = 50) -> List[AlertRecord]:
        """Retrieve recent alerts ordered by timestamp descending."""
        sql = "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?;"
        rows = self.db.execute_query(sql, (limit,))
        return [AlertRecord.from_dict(r) for r in rows]

    def get_alerts_between(self, start_time: str, end_time: str, limit: int = 100) -> List[AlertRecord]:
        """Query alerts occurring within a specified time range."""
        sql = """
        SELECT * FROM alerts
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC LIMIT ?;
        """
        rows = self.db.execute_query(sql, (to_iso8601(start_time), to_iso8601(end_time), limit))
        return [AlertRecord.from_dict(r) for r in rows]


class EvidenceRepository:
    """Data access for XAI factor contribution evidence."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def insert_evidence(self, evidence: EvidenceRecord) -> EvidenceRecord:
        """Insert single XAI factor evidence record."""
        sql = """
        INSERT OR IGNORE INTO evidence (
            evidence_id, event_id, alert_id, timestamp, factor,
            contribution, rank, statement, visual_available,
            model_explanation_available, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            evidence.evidence_id,
            evidence.event_id,
            evidence.alert_id,
            to_iso8601(evidence.timestamp),
            evidence.factor,
            evidence.contribution,
            evidence.rank,
            evidence.statement,
            serialize_bool(evidence.visual_available),
            serialize_bool(evidence.model_explanation_available),
            to_json_str(evidence.metadata_json),
        )
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if cursor.lastrowid:
                evidence.id = cursor.lastrowid
        return evidence

    def insert_evidence_batch(self, evidence_list: List[EvidenceRecord]) -> int:
        """Insert batch of XAI factor contributions in a single transaction."""
        if not evidence_list:
            return 0

        sql = """
        INSERT OR IGNORE INTO evidence (
            evidence_id, event_id, alert_id, timestamp, factor,
            contribution, rank, statement, visual_available,
            model_explanation_available, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        inserted = 0
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            for ev in evidence_list:
                params = (
                    ev.evidence_id,
                    ev.event_id,
                    ev.alert_id,
                    to_iso8601(ev.timestamp),
                    ev.factor,
                    ev.contribution,
                    ev.rank,
                    ev.statement,
                    serialize_bool(ev.visual_available),
                    serialize_bool(ev.model_explanation_available),
                    to_json_str(ev.metadata_json),
                )
                cursor.execute(sql, params)
                inserted += 1
        return inserted

    def get_evidence_for_event(self, event_id: str) -> List[EvidenceRecord]:
        """Fetch all XAI evidence items associated with an event_id."""
        sql = "SELECT * FROM evidence WHERE event_id = ? ORDER BY rank ASC, contribution DESC;"
        rows = self.db.execute_query(sql, (event_id,))
        return [EvidenceRecord.from_dict(r) for r in rows]

    def get_evidence_for_alert(self, alert_id: str) -> List[EvidenceRecord]:
        """Fetch all XAI evidence items associated with an alert_id."""
        sql = "SELECT * FROM evidence WHERE alert_id = ? ORDER BY rank ASC, contribution DESC;"
        rows = self.db.execute_query(sql, (alert_id,))
        return [EvidenceRecord.from_dict(r) for r in rows]


class SystemMetricsRepository:
    """Data access for sampled system operational telemetry."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def insert_metric(self, metric: SystemMetricRecord) -> SystemMetricRecord:
        """Insert sampled system metric row."""
        sql = """
        INSERT INTO system_metrics (
            session_id, timestamp, fps, frame_latency_ms,
            dashboard_latency_ms, people_count, crowd_density,
            anomaly_score, audio_score, fusion_score, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            metric.session_id,
            to_iso8601(metric.timestamp),
            metric.fps,
            metric.frame_latency_ms,
            metric.dashboard_latency_ms,
            metric.people_count,
            metric.crowd_density,
            metric.anomaly_score,
            metric.audio_score,
            metric.fusion_score,
            to_json_str(metric.metadata_json),
        )
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            metric.id = cursor.lastrowid
        return metric

    def get_recent_metrics(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[SystemMetricRecord]:
        """Retrieve recent metrics, optionally filtered by session_id."""
        if session_id:
            sql = "SELECT * FROM system_metrics WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?;"
            rows = self.db.execute_query(sql, (session_id, limit))
        else:
            sql = "SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT ?;"
            rows = self.db.execute_query(sql, (limit,))
        return [SystemMetricRecord.from_dict(r) for r in rows]


class HistoricalQueryAPI:
    """
    High-level facade uniting all repositories for historical analysis,
    summary telemetry statistics, and retention maintenance.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.sessions = SessionRepository(db)
        self.events = EventRepository(db)
        self.alerts = AlertRepository(db)
        self.evidence = EvidenceRepository(db)
        self.metrics = SystemMetricsRepository(db)

    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute historical summary statistics (counts, averages, peaks).
        These are descriptive database statistics, not ML evaluation metrics.
        """
        where_clause = "WHERE session_id = ?" if session_id else ""
        params = (session_id,) if session_id else ()

        sql = f"""
        SELECT
            count(*) as total_events,
            sum(CASE WHEN risk_state = 'EMERGENCY' THEN 1 ELSE 0 END) as emergency_count,
            sum(CASE WHEN risk_state = 'HIGH_RISK' THEN 1 ELSE 0 END) as high_risk_count,
            sum(CASE WHEN risk_state = 'SUSPICIOUS' THEN 1 ELSE 0 END) as suspicious_count,
            sum(CASE WHEN risk_state = 'NORMAL' THEN 1 ELSE 0 END) as normal_count,
            avg(anomaly_score) as avg_anomaly_score,
            max(anomaly_score) as max_anomaly_score,
            avg(crowd_density) as avg_crowd_density
        FROM events {where_clause};
        """
        rows = self.db.execute_query(sql, params)
        data = rows[0] if rows else {}

        sql_alerts = f"SELECT count(*) as total_alerts FROM alerts {where_clause};"
        alert_rows = self.db.execute_query(sql_alerts, params)
        total_alerts = int(alert_rows[0]["total_alerts"]) if alert_rows else 0

        return {
            "total_events": int(data.get("total_events") or 0),
            "total_alerts": total_alerts,
            "emergency_count": int(data.get("emergency_count") or 0),
            "high_risk_count": int(data.get("high_risk_count") or 0),
            "suspicious_count": int(data.get("suspicious_count") or 0),
            "normal_count": int(data.get("normal_count") or 0),
            "average_anomaly_score": round(float(data.get("avg_anomaly_score") or 0.0), 4),
            "maximum_anomaly_score": round(float(data.get("max_anomaly_score") or 0.0), 4),
            "average_crowd_density": round(float(data.get("avg_crowd_density") or 0.0), 3),
        }

    def prune_old_records(self, retention_days: int = 90) -> Dict[str, int]:
        """
        Prune records older than retention_days.
        Maintains referential integrity by deleting child tables before parents.
        """
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_iso = to_iso8601(cutoff_dt)

        pruned_counts = {
            "pruned_evidence": 0,
            "pruned_alerts": 0,
            "pruned_events": 0,
            "pruned_metrics": 0,
        }

        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # 1. Prune child evidence
            cursor.execute("DELETE FROM evidence WHERE timestamp < ?;", (cutoff_iso,))
            pruned_counts["pruned_evidence"] = cursor.rowcount

            # 2. Prune alerts
            cursor.execute("DELETE FROM alerts WHERE timestamp < ?;", (cutoff_iso,))
            pruned_counts["pruned_alerts"] = cursor.rowcount

            # 3. Prune events
            cursor.execute("DELETE FROM events WHERE timestamp < ?;", (cutoff_iso,))
            pruned_counts["pruned_events"] = cursor.rowcount

            # 4. Prune system metrics
            cursor.execute("DELETE FROM system_metrics WHERE timestamp < ?;", (cutoff_iso,))
            pruned_counts["pruned_metrics"] = cursor.rowcount

        logger.info(
            "Pruned database records older than %d days (%s): %s",
            retention_days,
            cutoff_iso,
            pruned_counts,
        )
        return pruned_counts
