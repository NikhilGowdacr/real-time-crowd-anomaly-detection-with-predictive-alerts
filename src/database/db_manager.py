"""
Database Connection and Transaction Manager - Phase 11.

Provides thread-safe connection handling, WAL mode, foreign key enforcement,
transaction context management (commit / rollback), and database health monitoring.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Generator, Optional, Union

from src.database.migrations import MigrationManager

logger = logging.getLogger("crowd_anomaly.database.manager")


class DatabaseManager:
    """
    Coordinates SQLite connections, migrations, transactions, and health checks.

    Guarantees:
    - Thread-safety: Thread-local connection pool avoids cross-thread SQLite sharing.
    - Foreign keys: Enforces PRAGMA foreign_keys = ON on every connection.
    - WAL Mode: Configures Write-Ahead Logging for high-concurrency read/write.
    - Busy timeout: Configures non-zero busy timeout to avoid lock collisions.
    - Corruption safety: Detects corruption without destructively deleting user data.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        wal_mode: bool = True,
        foreign_keys: bool = True,
        busy_timeout_ms: int = 5000,
        auto_create: bool = True,
    ) -> None:
        self.db_path_str = str(db_path) if db_path is not None else "data/crowd_anomaly.db"
        self.is_memory = (self.db_path_str == ":memory:" or self.db_path_str.startswith("file::memory:"))
        self.wal_mode = wal_mode and not self.is_memory
        self.foreign_keys = foreign_keys
        self.busy_timeout_ms = busy_timeout_ms
        self.auto_create = auto_create

        self._local = threading.local()
        self._all_connections: list = []
        self._lock = threading.Lock()
        self.last_write_at: Optional[str] = None

        if not self.is_memory and self.auto_create:
            # Ensure parent directory exists
            target_path = Path(self.db_path_str)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema and verify connectivity
        self._init_database()

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a fresh SQLite connection."""
        conn = sqlite3.connect(
            self.db_path_str,
            timeout=float(self.busy_timeout_ms) / 1000.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        if self.foreign_keys:
            cursor.execute("PRAGMA foreign_keys = ON;")

        if self.wal_mode:
            try:
                cursor.execute("PRAGMA journal_mode = WAL;")
            except Exception as e:
                logger.warning("Could not enable WAL mode: %s", e)

        cursor.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)};")
        cursor.close()

        with self._lock:
            self._all_connections.append(conn)

        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get or initialize thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn

    def _init_database(self) -> None:
        """Run schema migrations on initial startup."""
        try:
            conn = self.get_connection()
            MigrationManager.apply_migrations(conn)
        except Exception as e:
            logger.error("Database initialization failed: %s", e)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for transactional operations.
        Commits on normal exit, rolls back on exception.
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
            self.last_write_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            conn.rollback()
            logger.error("Transaction rolled back due to error: %s", exc)
            raise

    def execute_query(self, sql: str, params: tuple = ()) -> list:
        """Execute a read-only query and return matching rows as dictionaries."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute a single write statement within a transaction."""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.rowcount

    def health_check(self) -> Dict[str, Any]:
        """
        Examine database availability, table count, WAL status, and integrity.
        Never modifies or destroys corrupted database files.
        """
        res = {
            "available": False,
            "status": "OFFLINE",
            "path": self.db_path_str,
            "schema_version": 0,
            "table_count": 0,
            "wal_mode": "UNKNOWN",
            "last_write_at": self.last_write_at,
        }

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # 1. Quick integrity check
            cursor.execute("PRAGMA quick_check;")
            check_row = cursor.fetchone()
            if not check_row or check_row[0] != "ok":
                res["status"] = "DATABASE CORRUPTED"
                res["available"] = False
                logger.error("Database integrity check failed: %s", check_row)
                return res

            # 2. Schema version
            res["schema_version"] = MigrationManager.get_current_version(conn)

            # 3. Table count
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            t_row = cursor.fetchone()
            res["table_count"] = int(t_row[0]) if t_row else 0

            # 4. Journal mode
            cursor.execute("PRAGMA journal_mode;")
            jm_row = cursor.fetchone()
            res["wal_mode"] = str(jm_row[0]).upper() if jm_row else "UNKNOWN"

            res["available"] = True
            res["status"] = "ONLINE"
        except Exception as e:
            res["status"] = "OFFLINE"
            res["error"] = str(e)
            logger.warning("Database health check error: %s", e)

        return res

    def close(self) -> None:
        """Close all opened database connections."""
        with self._lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_connections.clear()
        if hasattr(self._local, "conn"):
            self._local.conn = None
