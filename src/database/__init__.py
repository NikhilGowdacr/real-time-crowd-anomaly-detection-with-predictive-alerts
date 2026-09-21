"""
Database and Persistence Module - Phase 11.

Provides local SQLite storage for sessions, decision events, alerts,
XAI factor evidence, and system metrics.
"""

from src.database.db_manager import DatabaseManager
from src.database.migrations import MigrationManager
from src.database.models import (
    AlertRecord,
    DBAlertRecord,
    DBEvidenceRecord,
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
    from_iso8601,
    from_json_str,
    to_iso8601,
    to_json_str,
)

__all__ = [
    "DatabaseManager",
    "MigrationManager",
    "SessionRecord",
    "EventRecord",
    "AlertRecord",
    "DBAlertRecord",
    "EvidenceRecord",
    "DBEvidenceRecord",
    "SystemMetricRecord",
    "SessionRepository",
    "EventRepository",
    "AlertRepository",
    "EvidenceRepository",
    "SystemMetricsRepository",
    "HistoricalQueryAPI",
    "SCHEMA_VERSION",
    "to_iso8601",
    "from_iso8601",
    "to_json_str",
    "from_json_str",
]
