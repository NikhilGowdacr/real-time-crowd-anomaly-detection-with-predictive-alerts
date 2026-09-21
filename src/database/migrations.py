"""
Database Migration and Schema Versioning System - Phase 11.

Applies additive, non-destructive schema migrations and records versioning
history in the schema_version table.
"""

from datetime import datetime, timezone
import logging
import sqlite3
from typing import Optional

from src.database.schema import (
    ALL_TABLE_STATEMENTS,
    CREATE_SCHEMA_VERSION_TABLE,
    INDEX_STATEMENTS,
    SCHEMA_VERSION,
)
from src.database.serialization import to_iso8601

logger = logging.getLogger("crowd_anomaly.database.migrations")


class MigrationManager:
    """
    Manages database schema initialization and forward migrations.
    Guarantees:
    - Non-destructive: Never calls DROP TABLE during normal migrations.
    - Idempotent: Can be safely called multiple times on the same database.
    - Audited: Records migration application timestamps.
    """

    @classmethod
    def get_current_version(cls, conn: sqlite3.Connection) -> int:
        """
        Check the currently applied schema version.
        Returns 0 if schema_version table does not exist or is empty.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version';"
            )
            if not cursor.fetchone():
                return 0

            cursor.execute("SELECT MAX(version) FROM schema_version;")
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
            return 0
        except Exception as e:
            logger.warning("Error checking schema version: %s", e)
            return 0

    @classmethod
    def apply_migrations(cls, conn: sqlite3.Connection) -> int:
        """
        Apply pending migrations up to SCHEMA_VERSION.
        Returns the active schema version.
        """
        current_version = cls.get_current_version(conn)
        target_version = SCHEMA_VERSION

        if current_version >= target_version:
            logger.debug("Database schema is up to date (version %d)", current_version)
            return current_version

        cursor = conn.cursor()

        # Step 1: Initial migration (Version 1)
        if current_version < 1:
            logger.info("Applying initial schema migration (Version 1)...")
            # Create schema_version table first
            cursor.execute(CREATE_SCHEMA_VERSION_TABLE)

            # Create all domain tables
            for ddl in ALL_TABLE_STATEMENTS:
                cursor.execute(ddl)

            # Create all indexes
            for idx in INDEX_STATEMENTS:
                cursor.execute(idx)

            # Record version
            applied_at = to_iso8601(datetime.now(timezone.utc))
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?);",
                (1, applied_at),
            )
            conn.commit()
            logger.info("Schema migration to Version 1 completed successfully.")
            current_version = 1

        # Placeholder for future additive migrations (Version 2, 3, etc.)
        # Example:
        # if current_version < 2:
        #     cursor.execute("ALTER TABLE events ADD COLUMN new_field TEXT;")
        #     cursor.execute("INSERT INTO schema_version ...")
        #     conn.commit()
        #     current_version = 2

        return current_version
