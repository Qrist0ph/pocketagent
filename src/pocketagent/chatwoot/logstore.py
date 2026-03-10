"""
SQLite-backed log store for Chatwoot callback traces.

Stores log records persistently, auto-purges entries older than 24 hours.
Each record captures the full decision path through the callback pipeline.

The database path is controlled by the LOGSTORE_DB_PATH environment variable
(default: /data/langlogs.db).
"""

import json
import os
import sqlite3
import time
import threading
import uuid
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("pocketagent.chatwoot.logstore")

RETENTION_SECONDS = 24 * 60 * 60  # 24 hours
DB_PATH = os.environ.get("LOGSTORE_DB_PATH", "/data/langlogs.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS log_records (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    conversation_id INTEGER,
    sender_email TEXT,
    content_preview TEXT,
    trace TEXT,
    intent TEXT,
    action TEXT,
    action_detail TEXT
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_log_records_timestamp ON log_records (timestamp)
"""


@dataclass
class LogRecord:
    id: str
    timestamp: float
    conversation_id: int | None
    sender_email: str
    content_preview: str
    trace: list[str]
    intent: str | None
    action: str | None
    action_detail: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "sender_email": self.sender_email,
            "content_preview": self.content_preview,
            "trace": self.trace,
            "intent": self.intent,
            "action": self.action,
            "action_detail": self.action_detail,
        }


class LogStore:
    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        except (OSError, ValueError):
            pass
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
                conn.commit()
            finally:
                conn.close()
        logger.info("LogStore initialized at %s", self._db_path)

    def create_record(
        self,
        conversation_id: int | None = None,
        sender_email: str = "",
        content_preview: str = "",
    ) -> LogRecord:
        record = LogRecord(
            id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            conversation_id=conversation_id,
            sender_email=sender_email,
            content_preview=content_preview[:500],
            trace=[],
            intent=None,
            action=None,
            action_detail=None,
        )
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO log_records (id, timestamp, conversation_id, sender_email, content_preview, trace, intent, action, action_detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (record.id, record.timestamp, record.conversation_id, record.sender_email, record.content_preview, json.dumps(record.trace), record.intent, record.action, record.action_detail),
                )
                conn.commit()
            finally:
                conn.close()
        self._purge()
        return record

    def save(self, record: LogRecord):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE log_records SET trace=?, intent=?, action=?, action_detail=? WHERE id=?",
                    (json.dumps(record.trace), record.intent, record.action, record.action_detail, record.id),
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, record_id: str) -> LogRecord | None:
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT id, timestamp, conversation_id, sender_email, content_preview, trace, intent, action, action_detail FROM log_records WHERE id=?", (record_id,)).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_all(self) -> list[LogRecord]:
        self._purge()
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("SELECT id, timestamp, conversation_id, sender_email, content_preview, trace, intent, action, action_detail FROM log_records ORDER BY timestamp DESC").fetchall()
            finally:
                conn.close()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row) -> LogRecord:
        return LogRecord(
            id=row[0],
            timestamp=row[1],
            conversation_id=row[2],
            sender_email=row[3],
            content_preview=row[4],
            trace=json.loads(row[5]) if row[5] else [],
            intent=row[6],
            action=row[7],
            action_detail=row[8],
        )

    def _purge(self):
        cutoff = time.time() - RETENTION_SECONDS
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM log_records WHERE timestamp < ?", (cutoff,))
                conn.commit()
            finally:
                conn.close()


# Singleton instance
log_store = LogStore()
