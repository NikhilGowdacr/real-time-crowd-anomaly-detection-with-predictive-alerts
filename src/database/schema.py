"""
Database Schema and DDL Definitions - Phase 11.

Defines SQLite schema, table structures, indexes, foreign key relationships,
and schema versioning for historical surveillance event persistence.
"""

SCHEMA_VERSION = 1

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    source_type TEXT,
    video_source TEXT,
    audio_enabled INTEGER DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    metadata_json TEXT
);
"""

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    session_id TEXT,
    timestamp TEXT NOT NULL,
    risk_state TEXT NOT NULL,
    anomaly_score REAL,
    decision_confidence REAL,
    alert_severity TEXT,
    dominant_modality TEXT,
    people_count INTEGER,
    crowd_density REAL,
    average_speed REAL,
    dispersion REAL,
    directional_divergence REAL,
    abnormal_track_count INTEGER,
    video_score REAL,
    audio_score REAL,
    fusion_score REAL,
    corroborated INTEGER,
    critical_evidence INTEGER,
    confirmed INTEGER,
    reason TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
"""

CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    event_id TEXT,
    session_id TEXT,
    timestamp TEXT NOT NULL,
    severity TEXT NOT NULL,
    risk_state TEXT NOT NULL,
    score REAL,
    confidence REAL,
    reason TEXT,
    dominant_modality TEXT,
    status TEXT NOT NULL DEFAULT 'CREATED',
    created_at TEXT NOT NULL,
    acknowledged_at TEXT,
    resolved_at TEXT,
    fingerprint TEXT,
    metadata_json TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
"""

CREATE_EVIDENCE_TABLE = """
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id TEXT UNIQUE NOT NULL,
    event_id TEXT,
    alert_id TEXT,
    timestamp TEXT NOT NULL,
    factor TEXT NOT NULL,
    contribution REAL NOT NULL,
    rank INTEGER,
    statement TEXT,
    visual_available INTEGER DEFAULT 1,
    model_explanation_available INTEGER DEFAULT 0,
    metadata_json TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE SET NULL
);
"""

CREATE_SYSTEM_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp TEXT NOT NULL,
    fps REAL,
    frame_latency_ms REAL,
    dashboard_latency_ms REAL,
    people_count INTEGER,
    crowd_density REAL,
    anomaly_score REAL,
    audio_score REAL,
    fusion_score REAL,
    metadata_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
"""

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_events_risk_state ON events(risk_state);",
    "CREATE INDEX IF NOT EXISTS idx_events_alert_severity ON events(alert_severity);",
    "CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_event_id ON alerts(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_session_id ON alerts(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_evidence_event_id ON evidence(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_evidence_alert_id ON evidence(alert_id);",
    "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_system_metrics_session_id ON system_metrics(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);",
]

ALL_TABLE_STATEMENTS = [
    CREATE_SCHEMA_VERSION_TABLE,
    CREATE_SESSIONS_TABLE,
    CREATE_EVENTS_TABLE,
    CREATE_ALERTS_TABLE,
    CREATE_EVIDENCE_TABLE,
    CREATE_SYSTEM_METRICS_TABLE,
]
