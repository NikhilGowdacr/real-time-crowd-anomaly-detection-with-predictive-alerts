"""
Unit tests for Phase 11: SQLite Database & Historical Event Persistence.

Covers:
1. Database initialization and automatic directory creation
2. Schema creation and schema_version tracking
3. Foreign key enforcement
4. WAL mode configuration
5. In-memory database behavior
6. Session CRUD lifecycle (create, get, list, end)
7. Event insertion and retrieval
8. Event recent query ordering
9. Event query by risk state
10. Event query by severity
11. Event query by time range
12. Alert insertion and retrieval
13. Alert active status query
14. Alert acknowledgement and resolution updates
15. Evidence batch insertion and query by event
16. Evidence query by alert
17. Referential relationships between session, event, alert, and evidence
18. ISO-8601 serialization and parsing
19. Boolean and enum serialization
20. JSON string serialization and deserialization
21. EventRecord dataclass serialization round-trip
22. AlertRecord dataclass serialization round-trip
23. EvidenceRecord dataclass serialization round-trip
24. SessionRecord dataclass serialization round-trip
25. Duplicate event idempotence (INSERT OR IGNORE)
26. Duplicate alert idempotence
27. Transaction atomic commit and rollback on error
28. Database health check reporting (ONLINE status)
29. Database corruption handling and reporting
30. System metrics persistence and query
31. Retention pruning with referential integrity (child before parent)
32. Concurrent multi-thread connection safety
33. Historical descriptive statistics aggregation
34. Immutability invariant (database operations do not mutate original objects)
35. Database manager graceful recovery and error safety
"""

from datetime import datetime, timedelta, timezone
import json
import sqlite3
import threading
import time
from typing import List
import pytest

from src.database.db_manager import DatabaseManager
from src.database.migrations import MigrationManager
from src.database.models import (
    AlertRecord,
    EventRecord,
    EvidenceRecord,
    SessionRecord,
    SystemMetricRecord,
)
from src.database.repositories import (
    AlertRepository,
    EventRepository,
    EvidenceRepository,
    HistoricalQueryAPI,
    SessionRepository,
    SystemMetricsRepository,
)
from src.database.schema import SCHEMA_VERSION
from src.database.serialization import (
    deserialize_bool,
    from_iso8601,
    from_json_str,
    serialize_bool,
    serialize_enum,
    to_iso8601,
    to_json_str,
)


# 1. Database Initialization & Auto Directory Creation
def test_db_initialization_and_auto_directory(tmp_path):
    nested_dir = tmp_path / "deep" / "nested" / "dir"
    db_file = nested_dir / "test_crowd.db"
    assert not nested_dir.exists()

    db = DatabaseManager(db_path=str(db_file), auto_create=True)
    assert nested_dir.exists()
    assert db_file.exists()

    health = db.health_check()
    assert health["status"] == "ONLINE"
    assert health["table_count"] >= 5
    db.close()


# 2. Schema Creation & Version Tracking
def test_schema_creation_and_version(tmp_path):
    db_file = tmp_path / "schema_test.db"
    db = DatabaseManager(db_path=str(db_file))

    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1;")
    row = cursor.fetchone()
    assert row is not None
    assert row["version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == 1

    # Verify tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {r["name"] for r in cursor.fetchall()}
    assert "schema_version" in tables
    assert "sessions" in tables
    assert "events" in tables
    assert "alerts" in tables
    assert "evidence" in tables
    assert "system_metrics" in tables
    db.close()


# 3. Foreign Key Enforcement
def test_foreign_keys_enforced(tmp_path):
    db_file = tmp_path / "fk_test.db"
    db = DatabaseManager(db_path=str(db_file), foreign_keys=True)

    # Inserting event with non-existent session_id should violate foreign key constraint
    invalid_event = EventRecord(
        event_id="EVT-BAD-FK",
        session_id="NON_EXISTENT_SESSION",
        timestamp=datetime.now(timezone.utc).isoformat(),
        risk_state="NORMAL",
        anomaly_score=0.1,
    )
    repo = EventRepository(db)
    with pytest.raises(sqlite3.IntegrityError):
        repo.insert_event(invalid_event)

    db.close()


# 4. WAL Mode Enabled
def test_wal_mode_enabled(tmp_path):
    db_file = tmp_path / "wal_test.db"
    db = DatabaseManager(db_path=str(db_file), wal_mode=True)
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal"
    db.close()


# 5. In-Memory DB Skips WAL Gracefully
def test_in_memory_db_skips_wal():
    db = DatabaseManager(db_path=":memory:", wal_mode=True)
    health = db.health_check()
    assert health["status"] == "ONLINE"
    assert health["path"] == ":memory:"
    assert health["wal_mode"] != "WAL"
    db.close()


# 6. Session CRUD Lifecycle
def test_session_crud_lifecycle(tmp_path):
    db_file = tmp_path / "session_test.db"
    db = DatabaseManager(db_path=str(db_file))
    repo = SessionRepository(db)

    # Create
    t_start = datetime.now(timezone.utc)
    sess = SessionRecord(
        session_id="SESS-001",
        started_at=t_start.isoformat(),
        source_type="synthetic",
        video_source="synthetic_stream",
        audio_enabled=True,
        status="ACTIVE",
        metadata_json={"operator": "admin"},
    )
    repo.create_session(sess)

    # Get
    fetched = repo.get_session("SESS-001")
    assert fetched is not None
    assert fetched.session_id == "SESS-001"
    assert fetched.source_type == "synthetic"
    assert fetched.audio_enabled is True
    assert fetched.status == "ACTIVE"
    assert fetched.metadata_json == {"operator": "admin"}

    # End
    t_end = t_start + timedelta(minutes=10)
    updated = repo.end_session("SESS-001", status="COMPLETED", ended_at=t_end.isoformat())
    assert updated is True

    fetched_ended = repo.get_session("SESS-001")
    assert fetched_ended.status == "COMPLETED"
    assert fetched_ended.ended_at is not None

    # List
    sessions = repo.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == "SESS-001"
    db.close()


# 7. Event Insertion and Retrieval
def test_event_insertion_and_get(tmp_path):
    db_file = tmp_path / "event_test.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)

    sess = SessionRecord(session_id="SESS-EV-1")
    sess_repo.create_session(sess)

    evt = EventRecord(
        event_id="EVT-001",
        session_id="SESS-EV-1",
        timestamp="2026-03-10T12:00:00Z",
        risk_state="HIGH_RISK",
        anomaly_score=0.78,
        decision_confidence=0.92,
        alert_severity="HIGH",
        dominant_modality="video",
        people_count=15,
        crowd_density=0.45,
        average_speed=12.5,
        video_score=0.82,
        audio_score=0.35,
        fusion_score=0.78,
        corroborated=False,
        critical_evidence=False,
        confirmed=True,
        reason="Rapid scattering observed in central sector",
        created_at="2026-03-10T12:00:00Z",
        metadata_json={"zone": "sector_A"},
    )
    ev_repo.insert_event(evt)

    retrieved = ev_repo.get_event("EVT-001")
    assert retrieved is not None
    assert retrieved.event_id == "EVT-001"
    assert retrieved.risk_state == "HIGH_RISK"
    assert abs(retrieved.anomaly_score - 0.78) < 1e-4
    assert retrieved.people_count == 15
    assert retrieved.corroborated is False
    assert retrieved.confirmed is True
    assert retrieved.metadata_json == {"zone": "sector_A"}
    db.close()


# 8. Event Query Recent Ordering
def test_event_query_recent(tmp_path):
    db_file = tmp_path / "event_recent.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-REC"))

    t0 = datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        t_iso = (t0 + timedelta(seconds=i * 10)).isoformat()
        ev_repo.insert_event(
            EventRecord(
                event_id=f"EVT-R-{i}",
                session_id="SESS-REC",
                timestamp=t_iso,
                risk_state="NORMAL",
                anomaly_score=0.1 * i,
            )
        )

    recent = ev_repo.get_recent_events(limit=3)
    assert len(recent) == 3
    # Ordered descending: newest first
    assert recent[0].event_id == "EVT-R-4"
    assert recent[1].event_id == "EVT-R-3"
    assert recent[2].event_id == "EVT-R-2"
    db.close()


# 9. Event Query by Risk State
def test_event_query_by_state(tmp_path):
    db_file = tmp_path / "event_state.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-ST"))

    states = ["NORMAL", "NORMAL", "SUSPICIOUS", "EMERGENCY", "EMERGENCY"]
    for i, st in enumerate(states):
        ev_repo.insert_event(
            EventRecord(
                event_id=f"EVT-ST-{i}",
                session_id="SESS-ST",
                timestamp=f"2026-03-10T11:00:0{i}Z",
                risk_state=st,
                anomaly_score=0.2 * i,
            )
        )

    emergencies = ev_repo.query_events_by_state("EMERGENCY")
    assert len(emergencies) == 2
    for e in emergencies:
        assert e.risk_state == "EMERGENCY"

    normals = ev_repo.query_events_by_state("NORMAL")
    assert len(normals) == 2
    db.close()


# 10. Event Query by Severity
def test_event_query_by_severity(tmp_path):
    db_file = tmp_path / "event_sev.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-SEV"))

    ev_repo.insert_event(EventRecord(event_id="E-1", session_id="SESS-SEV", timestamp="2026-03-10T10:00:00Z", alert_severity="INFO"))
    ev_repo.insert_event(EventRecord(event_id="E-2", session_id="SESS-SEV", timestamp="2026-03-10T10:01:00Z", alert_severity="CRITICAL"))
    ev_repo.insert_event(EventRecord(event_id="E-3", session_id="SESS-SEV", timestamp="2026-03-10T10:02:00Z", alert_severity="CRITICAL"))

    crits = ev_repo.query_events_by_severity("CRITICAL")
    assert len(crits) == 2
    for c in crits:
        assert c.alert_severity == "CRITICAL"
    db.close()


# 11. Event Query by Time Range
def test_event_query_by_time_range(tmp_path):
    db_file = tmp_path / "event_timerange.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-TR"))

    t_times = [
        "2026-03-10T08:00:00Z",
        "2026-03-10T08:30:00Z",
        "2026-03-10T09:00:00Z",
        "2026-03-10T09:30:00Z",
        "2026-03-10T10:00:00Z",
    ]
    for i, t in enumerate(t_times):
        ev_repo.insert_event(EventRecord(event_id=f"EVT-TR-{i}", session_id="SESS-TR", timestamp=t))

    ranged = ev_repo.query_events_by_time_range("2026-03-10T08:15:00Z", "2026-03-10T09:45:00Z")
    assert len(ranged) == 3
    event_ids = [e.event_id for e in ranged]
    assert "EVT-TR-1" in event_ids
    assert "EVT-TR-2" in event_ids
    assert "EVT-TR-3" in event_ids
    db.close()


# 12. Alert Insertion and Retrieval
def test_alert_insertion_and_get(tmp_path):
    db_file = tmp_path / "alert_test.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)
    alt_repo = AlertRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-ALT"))
    ev_repo.insert_event(EventRecord(event_id="EVT-A-1", session_id="SESS-ALT", timestamp="2026-03-10T12:00:00Z"))

    alert = AlertRecord(
        alert_id="ALT-001",
        event_id="EVT-A-1",
        session_id="SESS-ALT",
        timestamp="2026-03-10T12:00:00Z",
        severity="CRITICAL",
        risk_state="EMERGENCY",
        score=0.95,
        confidence=0.98,
        reason="Firearm detected",
        dominant_modality="video",
        status="ACTIVE",
        fingerprint="fp_weapon_001",
        metadata_json={"location": "gate_1"},
    )
    alt_repo.insert_alert(alert)

    retrieved = alt_repo.get_alert("ALT-001")
    assert retrieved is not None
    assert retrieved.alert_id == "ALT-001"
    assert retrieved.severity == "CRITICAL"
    assert retrieved.status == "ACTIVE"
    assert retrieved.fingerprint == "fp_weapon_001"
    assert retrieved.metadata_json == {"location": "gate_1"}
    db.close()


# 13. Alert Active Query
def test_alert_query_active(tmp_path):
    db_file = tmp_path / "alert_active.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)
    alt_repo = AlertRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-ACT"))
    ev_repo.insert_event(EventRecord(event_id="EVT-ACT-1", session_id="SESS-ACT", timestamp="2026-03-10T12:00:00Z"))

    alt_repo.insert_alert(AlertRecord(alert_id="A-1", event_id="EVT-ACT-1", session_id="SESS-ACT", status="ACTIVE"))
    alt_repo.insert_alert(AlertRecord(alert_id="A-2", event_id="EVT-ACT-1", session_id="SESS-ACT", status="ACKNOWLEDGED"))
    alt_repo.insert_alert(AlertRecord(alert_id="A-3", event_id="EVT-ACT-1", session_id="SESS-ACT", status="RESOLVED"))

    actives = alt_repo.get_active_alerts()
    assert len(actives) == 1
    assert actives[0].alert_id == "A-1"
    db.close()


# 14. Alert Acknowledgement and Resolution Updates
def test_alert_acknowledgement_and_resolution(tmp_path):
    db_file = tmp_path / "alert_lifecycle.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)
    alt_repo = AlertRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-LC"))
    ev_repo.insert_event(EventRecord(event_id="EVT-LC-1", session_id="SESS-LC", timestamp="2026-03-10T12:00:00Z"))

    alt_repo.insert_alert(
        AlertRecord(
            alert_id="ALT-LC",
            event_id="EVT-LC-1",
            session_id="SESS-LC",
            status="ACTIVE",
        )
    )

    # Acknowledge
    t_ack = "2026-03-10T12:01:00Z"
    alt_repo.update_alert_status("ALT-LC", status="ACKNOWLEDGED", acknowledged_at=t_ack)
    ack_alt = alt_repo.get_alert("ALT-LC")
    assert ack_alt.status == "ACKNOWLEDGED"
    assert ack_alt.acknowledged_at == t_ack
    assert ack_alt.resolved_at is None

    # Resolve
    t_res = "2026-03-10T12:05:00Z"
    alt_repo.update_alert_status("ALT-LC", status="RESOLVED", resolved_at=t_res)
    res_alt = alt_repo.get_alert("ALT-LC")
    assert res_alt.status == "RESOLVED"
    assert res_alt.resolved_at == t_res
    db.close()


# 15. Evidence Batch Insertion and Query by Event
def test_evidence_batch_insertion_and_query_by_event(tmp_path):
    db_file = tmp_path / "evidence_test.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)
    evi_repo = EvidenceRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-EV-E"))
    ev_repo.insert_event(EventRecord(event_id="EVT-E-1", session_id="SESS-EV-E", timestamp="2026-03-10T12:00:00Z"))

    ev1 = EvidenceRecord(
        evidence_id="EVID-1",
        event_id="EVT-E-1",
        factor="weapon_presence",
        contribution=0.45,
        rank=1,
        statement="Firearm visible",
    )
    ev2 = EvidenceRecord(
        evidence_id="EVID-2",
        event_id="EVT-E-1",
        factor="acoustic_scream",
        contribution=0.30,
        rank=2,
        statement="Scream detected",
    )
    evi_repo.insert_evidence_batch([ev1, ev2])

    items = evi_repo.get_evidence_for_event("EVT-E-1")
    assert len(items) == 2
    assert items[0].factor == "weapon_presence"
    assert items[0].rank == 1
    assert items[1].factor == "acoustic_scream"
    assert items[1].rank == 2
    db.close()


# 16. Evidence Query by Alert
def test_evidence_query_by_alert(tmp_path):
    db_file = tmp_path / "evidence_alert.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)
    alt_repo = AlertRepository(db)
    evi_repo = EvidenceRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-EA"))
    ev_repo.insert_event(EventRecord(event_id="EVT-EA-1", session_id="SESS-EA", timestamp="2026-03-10T12:00:00Z"))
    alt_repo.insert_alert(AlertRecord(alert_id="ALT-EA-1", event_id="EVT-EA-1", session_id="SESS-EA"))

    ev = EvidenceRecord(
        evidence_id="EVID-EA-1",
        event_id="EVT-EA-1",
        alert_id="ALT-EA-1",
        factor="crowd_turbulence",
        contribution=0.55,
        rank=1,
    )
    evi_repo.insert_evidence_batch([ev])

    items = evi_repo.get_evidence_for_alert("ALT-EA-1")
    assert len(items) == 1
    assert items[0].evidence_id == "EVID-EA-1"
    assert items[0].factor == "crowd_turbulence"
    db.close()


# 17. Referential Integrity Linkages
def test_event_alert_evidence_relationship(tmp_path):
    db_file = tmp_path / "referential.db"
    db = DatabaseManager(db_path=str(db_file), foreign_keys=True)
    api = HistoricalQueryAPI(db)

    # 1. Create Session
    sess = SessionRecord(session_id="SESS-REF-1")
    api.sessions.create_session(sess)

    # 2. Create Event linked to Session
    evt = EventRecord(event_id="EVT-REF-1", session_id="SESS-REF-1")
    api.events.insert_event(evt)

    # 3. Create Alert linked to Event and Session
    alt = AlertRecord(alert_id="ALT-REF-1", event_id="EVT-REF-1", session_id="SESS-REF-1")
    api.alerts.insert_alert(alt)

    # 4. Create Evidence linked to Event and Alert
    ev = EvidenceRecord(evidence_id="EVI-REF-1", event_id="EVT-REF-1", alert_id="ALT-REF-1")
    api.evidence.insert_evidence_batch([ev])

    # Verify retrieval links
    linked_alerts = api.alerts.get_alerts_for_event("EVT-REF-1")
    assert len(linked_alerts) == 1
    assert linked_alerts[0].alert_id == "ALT-REF-1"

    linked_ev = api.evidence.get_evidence_for_event("EVT-REF-1")
    assert len(linked_ev) == 1
    assert linked_ev[0].alert_id == "ALT-REF-1"
    db.close()


# 18. ISO-8601 Serialization and Parsing
def test_serialization_iso8601():
    # None handling
    assert to_iso8601(None) is None
    assert from_iso8601(None) is None

    # Epoch float
    epoch = 1700000000.0
    iso_str = to_iso8601(epoch)
    assert isinstance(iso_str, str)
    parsed_epoch = from_iso8601(iso_str)
    assert parsed_epoch is not None
    assert abs(parsed_epoch - epoch) < 1.0

    # Datetime instance
    dt = datetime(2026, 3, 10, 15, 30, 45, tzinfo=timezone.utc)
    assert to_iso8601(dt) == "2026-03-10T15:30:45+00:00"


# 19. Boolean and Enum Serialization
def test_serialization_booleans_and_enums():
    assert serialize_bool(True) == 1
    assert serialize_bool(False) == 0
    assert serialize_bool(None) is None
    assert deserialize_bool(1) is True
    assert deserialize_bool(0) is False
    assert deserialize_bool(None) is None

    from enum import Enum

    class TestEnum(Enum):
        ALPHA = "alpha_val"
        BETA = "beta_val"

    assert serialize_enum(TestEnum.ALPHA) == "alpha_val"
    assert serialize_enum("raw_string") == "raw_string"
    assert serialize_enum(None) is None


# 20. JSON String Serialization & Deserialization
def test_serialization_json():
    data = {"key": "value", "count": 42, "flags": [True, False]}
    s = to_json_str(data)
    assert isinstance(s, str)
    parsed = from_json_str(s)
    assert parsed == data

    assert to_json_str(None) is None
    assert from_json_str(None) == {}
    assert from_json_str("invalid json") == {}


# 21. EventRecord Serialization Round-Trip
def test_event_dataclass_serialization():
    rec = EventRecord(
        event_id="EVT-SER-1",
        session_id="SESS-1",
        risk_state="EMERGENCY",
        anomaly_score=0.92,
        dominant_modality="audio",
        corroborated=True,
        metadata_json={"alert_rule": "crit_scream"},
    )
    d = rec.to_dict()
    assert d["event_id"] == "EVT-SER-1"
    assert d["risk_state"] == "EMERGENCY"
    assert d["corroborated"] is True

    json_str = rec.to_json()
    reconstructed = EventRecord.from_json(json_str)
    assert reconstructed.event_id == rec.event_id
    assert reconstructed.anomaly_score == rec.anomaly_score
    assert reconstructed.corroborated is True
    assert reconstructed.metadata_json == {"alert_rule": "crit_scream"}


# 22. AlertRecord Serialization Round-Trip
def test_alert_dataclass_serialization():
    rec = AlertRecord(
        alert_id="ALT-SER-1",
        event_id="EVT-1",
        session_id="SESS-1",
        severity="CRITICAL",
        risk_state="EMERGENCY",
        score=0.95,
        status="ACTIVE",
    )
    d = rec.to_dict()
    assert d["alert_id"] == "ALT-SER-1"
    assert d["severity"] == "CRITICAL"

    json_str = rec.to_json()
    reconstructed = AlertRecord.from_json(json_str)
    assert reconstructed.alert_id == rec.alert_id
    assert reconstructed.score == rec.score
    assert reconstructed.status == "ACTIVE"


# 23. EvidenceRecord Serialization Round-Trip
def test_evidence_dataclass_serialization():
    rec = EvidenceRecord(
        evidence_id="EVID-SER-1",
        event_id="EVT-1",
        factor="weapon_presence",
        contribution=0.48,
        rank=1,
    )
    d = rec.to_dict()
    assert d["evidence_id"] == "EVID-SER-1"
    assert d["rank"] == 1

    json_str = rec.to_json()
    reconstructed = EvidenceRecord.from_json(json_str)
    assert reconstructed.evidence_id == rec.evidence_id
    assert reconstructed.factor == "weapon_presence"
    assert reconstructed.contribution == 0.48


# 24. SessionRecord Serialization Round-Trip
def test_session_dataclass_serialization():
    rec = SessionRecord(
        session_id="SESS-SER-1",
        source_type="rtsp",
        video_source="rtsp://cam1:554/live",
        audio_enabled=False,
        status="ACTIVE",
    )
    d = rec.to_dict()
    assert d["session_id"] == "SESS-SER-1"
    assert d["audio_enabled"] is False

    json_str = rec.to_json()
    reconstructed = SessionRecord.from_json(json_str)
    assert reconstructed.session_id == rec.session_id
    assert reconstructed.source_type == "rtsp"
    assert reconstructed.audio_enabled is False


# 25. Duplicate Event Idempotency
def test_duplicate_event_idempotency(tmp_path):
    db_file = tmp_path / "dup_event.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-DUP-E"))
    evt = EventRecord(event_id="EVT-DUP-1", session_id="SESS-DUP-E", timestamp="2026-03-10T12:00:00Z", anomaly_score=0.5)

    # Insert once
    ev_repo.insert_event(evt)
    # Insert again with same event_id
    ev_repo.insert_event(evt)

    events = ev_repo.get_recent_events(limit=10)
    assert len(events) == 1
    db.close()


# 26. Duplicate Alert Idempotency
def test_duplicate_alert_idempotency(tmp_path):
    db_file = tmp_path / "dup_alert.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)
    alt_repo = AlertRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-DUP-A"))
    ev_repo.insert_event(EventRecord(event_id="EVT-DUP-A1", session_id="SESS-DUP-A"))

    alt = AlertRecord(alert_id="ALT-DUP-1", event_id="EVT-DUP-A1", session_id="SESS-DUP-A")
    alt_repo.insert_alert(alt)
    alt_repo.insert_alert(alt)

    alerts = alt_repo.get_recent_alerts(limit=10)
    assert len(alerts) == 1
    db.close()


# 27. Transaction Rollback on Exception
def test_transaction_atomic_rollback(tmp_path):
    db_file = tmp_path / "tx_rollback.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)

    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, started_at, status) VALUES ('SESS-TX-FAIL', '2026-03-10T12:00:00Z', 'ACTIVE');"
            )
            # Deliberate failure inside transaction
            raise RuntimeError("Simulated transaction fault")

    # Verify record was rolled back
    sess = sess_repo.get_session("SESS-TX-FAIL")
    assert sess is None
    db.close()


# 28. Database Health Check Online
def test_health_check_online(tmp_path):
    db_file = tmp_path / "health_online.db"
    db = DatabaseManager(db_path=str(db_file))
    health = db.health_check()
    assert health["status"] == "ONLINE"
    assert health["schema_version"] == 1
    assert health["table_count"] >= 5
    assert health["available"] is True
    db.close()


# 29. Database Corruption Reporting
def test_health_check_corrupted(tmp_path):
    db_file = tmp_path / "health_corrupt.db"
    db = DatabaseManager(db_path=str(db_file))
    db.close()

    # Corrupt the SQLite header bytes
    with open(db_file, "r+b") as f:
        f.seek(0)
        f.write(b"CORRUPTED_GARBAGE_HEADER_DATA_NOT_SQLITE")

    # DatabaseManager should detect corruption safely
    db_corrupted = DatabaseManager(db_path=str(db_file), auto_create=False)
    health = db_corrupted.health_check()
    assert health["status"] in ("DATABASE CORRUPTED", "OFFLINE")
    db_corrupted.close()


# 30. System Metrics Persistence and Query
def test_system_metrics_persistence_and_query(tmp_path):
    db_file = tmp_path / "metrics_test.db"
    db = DatabaseManager(db_path=str(db_file))
    sess_repo = SessionRepository(db)
    metrics_repo = SystemMetricsRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-MET"))

    m1 = SystemMetricRecord(session_id="SESS-MET", fps=29.5, frame_latency_ms=18.2, people_count=10, crowd_density=0.25)
    m2 = SystemMetricRecord(session_id="SESS-MET", fps=30.0, frame_latency_ms=16.5, people_count=12, crowd_density=0.30)
    metrics_repo.insert_metric(m1)
    metrics_repo.insert_metric(m2)

    recent = metrics_repo.get_recent_metrics(session_id="SESS-MET", limit=10)
    assert len(recent) == 2
    assert recent[0].fps in (29.5, 30.0)
    db.close()


# 31. Retention Pruning with Referential Integrity
def test_retention_pruning_referential_integrity(tmp_path):
    db_file = tmp_path / "retention_test.db"
    db = DatabaseManager(db_path=str(db_file), foreign_keys=True)
    api = HistoricalQueryAPI(db)

    sess = SessionRecord(session_id="SESS-RET")
    api.sessions.create_session(sess)

    t_now = datetime.now(timezone.utc)
    t_old = (t_now - timedelta(days=120)).isoformat()
    t_new = (t_now - timedelta(days=5)).isoformat()

    # Insert OLD event, alert, evidence
    api.events.insert_event(EventRecord(event_id="EVT-OLD", session_id="SESS-RET", timestamp=t_old))
    api.alerts.insert_alert(AlertRecord(alert_id="ALT-OLD", event_id="EVT-OLD", session_id="SESS-RET", timestamp=t_old))
    api.evidence.insert_evidence_batch([EvidenceRecord(evidence_id="EVI-OLD", event_id="EVT-OLD", alert_id="ALT-OLD", timestamp=t_old)])

    # Insert NEW event, alert, evidence
    api.events.insert_event(EventRecord(event_id="EVT-NEW", session_id="SESS-RET", timestamp=t_new))
    api.alerts.insert_alert(AlertRecord(alert_id="ALT-NEW", event_id="EVT-NEW", session_id="SESS-RET", timestamp=t_new))
    api.evidence.insert_evidence_batch([EvidenceRecord(evidence_id="EVI-NEW", event_id="EVT-NEW", alert_id="ALT-NEW", timestamp=t_new)])

    # Prune older than 90 days
    pruned = api.prune_old_records(retention_days=90)
    assert pruned["pruned_events"] == 1
    assert pruned["pruned_alerts"] == 1
    assert pruned["pruned_evidence"] == 1

    # Verify OLD records are deleted and NEW records are preserved
    assert api.events.get_event("EVT-OLD") is None
    assert api.alerts.get_alert("ALT-OLD") is None
    assert len(api.evidence.get_evidence_for_event("EVT-OLD")) == 0

    assert api.events.get_event("EVT-NEW") is not None
    assert api.alerts.get_alert("ALT-NEW") is not None
    assert len(api.evidence.get_evidence_for_event("EVT-NEW")) == 1
    db.close()


# 32. Concurrent Thread Access Safety
def test_concurrent_threads_database_access(tmp_path):
    db_file = tmp_path / "concurrent_test.db"
    db = DatabaseManager(db_path=str(db_file), wal_mode=True, busy_timeout_ms=5000)
    sess_repo = SessionRepository(db)
    ev_repo = EventRepository(db)

    sess_repo.create_session(SessionRecord(session_id="SESS-CONC"))

    thread_errors: List[Exception] = []

    def worker_insert(worker_id: int):
        try:
            for i in range(10):
                eid = f"EVT-W{worker_id}-{i}"
                ev_repo.insert_event(
                    EventRecord(
                        event_id=eid,
                        session_id="SESS-CONC",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        anomaly_score=0.1,
                    )
                )
        except Exception as exc:
            thread_errors.append(exc)

    threads = [threading.Thread(target=worker_insert, args=(t_id,)) for t_id in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(thread_errors) == 0
    all_events = ev_repo.get_recent_events(limit=100)
    assert len(all_events) == 40
    db.close()


# 33. Historical Statistics Aggregation
def test_historical_statistics_aggregation(tmp_path):
    db_file = tmp_path / "stats_test.db"
    db = DatabaseManager(db_path=str(db_file))
    api = HistoricalQueryAPI(db)

    api.sessions.create_session(SessionRecord(session_id="SESS-STAT"))

    # 3 NORMAL, 2 HIGH_RISK, 1 EMERGENCY
    dataset = [
        ("EV-S-1", "NORMAL", 0.10, 0.2),
        ("EV-S-2", "NORMAL", 0.15, 0.3),
        ("EV-S-3", "NORMAL", 0.20, 0.4),
        ("EV-S-4", "HIGH_RISK", 0.70, 0.8),
        ("EV-S-5", "HIGH_RISK", 0.80, 0.9),
        ("EV-S-6", "EMERGENCY", 0.95, 1.2),
    ]
    for eid, state, score, density in dataset:
        api.events.insert_event(
            EventRecord(
                event_id=eid,
                session_id="SESS-STAT",
                risk_state=state,
                anomaly_score=score,
                crowd_density=density,
            )
        )

    # 2 Alerts
    api.alerts.insert_alert(AlertRecord(alert_id="ALT-S-1", event_id="EV-S-5", session_id="SESS-STAT"))
    api.alerts.insert_alert(AlertRecord(alert_id="ALT-S-2", event_id="EV-S-6", session_id="SESS-STAT"))

    stats = api.get_statistics()
    assert stats["total_events"] == 6
    assert stats["total_alerts"] == 2
    assert stats["emergency_count"] == 1
    assert stats["high_risk_count"] == 2
    assert stats["normal_count"] == 3
    assert stats["maximum_anomaly_score"] == 0.95
    expected_avg = round((0.10 + 0.15 + 0.20 + 0.70 + 0.80 + 0.95) / 6, 4)
    assert stats["average_anomaly_score"] == expected_avg
    db.close()


# 34. Non-Mutation Invariant
def test_non_mutation_invariant(tmp_path):
    db_file = tmp_path / "immutability.db"
    db = DatabaseManager(db_path=str(db_file))
    api = HistoricalQueryAPI(db)

    api.sessions.create_session(SessionRecord(session_id="SESS-IMM"))

    original_score = 0.88
    original_state = "HIGH_RISK"
    evt = EventRecord(
        event_id="EVT-IMM-1",
        session_id="SESS-IMM",
        risk_state=original_state,
        anomaly_score=original_score,
        decision_confidence=0.91,
    )

    api.events.insert_event(evt)
    stats = api.get_statistics()
    queried = api.events.get_event("EVT-IMM-1")

    # Ingestion, statistics, and queries must NEVER mutate the caller's event object
    assert evt.anomaly_score == original_score
    assert evt.risk_state == original_state
    assert queried.anomaly_score == original_score
    assert queried.risk_state == original_state
    db.close()


# 35. Missing Database Directory Recovery
def test_missing_database_directory_auto_created(tmp_path):
    new_dir = tmp_path / "brand_new_subfolder"
    target_file = new_dir / "crowd.db"
    assert not new_dir.exists()

    db = DatabaseManager(db_path=str(target_file), auto_create=True)
    assert new_dir.exists()
    assert target_file.exists()
    health = db.health_check()
    assert health["status"] == "ONLINE"
    db.close()
