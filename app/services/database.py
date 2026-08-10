"""Small optional persistence layer backed by DATABASE_URL.

The application remains usable without a database. When DATABASE_URL is set,
JSON state and binary analysis artifacts are stored in one automatically-created
table. File storage remains the recovery fallback.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone


logger = logging.getLogger(__name__)
_engine = None
_table = None
_initialization_lock = threading.Lock()
_initialization_attempted = False
_last_initialization_failure = 0.0


def configured():
    return bool(os.getenv("DATABASE_URL"))


def _normalized_url():
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    if url.startswith("mysql://"):
        return "mysql+pymysql://" + url[len("mysql://"):]
    return url


def _initialize():
    global _engine, _table, _initialization_attempted, _last_initialization_failure
    retry_seconds = max(5, int(os.getenv("DATABASE_RETRY_SECONDS", "60")))
    if _initialization_attempted and (_engine is not None or time.monotonic() - _last_initialization_failure < retry_seconds):
        return _engine, _table
    with _initialization_lock:
        if _initialization_attempted and (_engine is not None or time.monotonic() - _last_initialization_failure < retry_seconds):
            return _engine, _table
        _initialization_attempted = True
        if not configured():
            return None, None
        try:
            from sqlalchemy import Column, DateTime, LargeBinary, MetaData, String, Table, Text, create_engine

            _engine = create_engine(_normalized_url(), pool_pre_ping=True, future=True)
            metadata = MetaData()
            _table = Table(
                "kabumikke_store",
                metadata,
                Column("namespace", String(64), primary_key=True),
                Column("item_key", String(255), primary_key=True),
                Column("payload_json", Text, nullable=True),
                Column("payload_binary", LargeBinary, nullable=True),
                Column("updated_at", DateTime(timezone=True), nullable=False),
            )
            metadata.create_all(_engine)
        except Exception:
            logger.exception("database initialization failed; file storage will be used")
            _engine, _table = None, None
            _last_initialization_failure = time.monotonic()
        return _engine, _table


def _put(namespace, key, json_payload=None, binary_payload=None):
    engine, table = _initialize()
    if engine is None:
        return False
    try:
        with engine.begin() as connection:
            values = {
                "namespace": namespace,
                "item_key": key,
                "payload_json": json_payload,
                "payload_binary": binary_payload,
                "updated_at": datetime.now(timezone.utc),
            }
            if engine.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert
                statement = insert(table).values(**values).on_conflict_do_update(
                    index_elements=[table.c.namespace, table.c.item_key],
                    set_={name: value for name, value in values.items() if name not in {"namespace", "item_key"}},
                )
                connection.execute(statement)
            elif engine.dialect.name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert
                statement = insert(table).values(**values).on_conflict_do_update(
                    index_elements=[table.c.namespace, table.c.item_key],
                    set_={name: value for name, value in values.items() if name not in {"namespace", "item_key"}},
                )
                connection.execute(statement)
            elif engine.dialect.name in {"mysql", "mariadb"}:
                from sqlalchemy.dialects.mysql import insert
                statement = insert(table).values(**values)
                statement = statement.on_duplicate_key_update(
                    payload_json=statement.inserted.payload_json,
                    payload_binary=statement.inserted.payload_binary,
                    updated_at=statement.inserted.updated_at,
                )
                connection.execute(statement)
            else:
                from sqlalchemy import and_, insert, update
                updated = connection.execute(
                    update(table).where(
                        and_(table.c.namespace == namespace, table.c.item_key == key)
                    ).values(**{name: value for name, value in values.items() if name not in {"namespace", "item_key"}})
                )
                if not updated.rowcount:
                    connection.execute(insert(table).values(**values))
        return True
    except Exception:
        logger.exception("database write failed: namespace=%s key=%s", namespace, key)
        return False


def put_json(namespace, key, payload):
    try:
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        logger.exception("database JSON serialization failed: namespace=%s key=%s", namespace, key)
        return False
    return _put(namespace, key, json_payload=serialized)


def get_json(namespace, key):
    engine, table = _initialize()
    if engine is None:
        return None
    try:
        from sqlalchemy import and_, select

        with engine.connect() as connection:
            value = connection.execute(
                select(table.c.payload_json).where(
                    and_(table.c.namespace == namespace, table.c.item_key == key)
                )
            ).scalar_one_or_none()
        return json.loads(value) if value is not None else None
    except Exception:
        logger.exception("database read failed: namespace=%s key=%s", namespace, key)
        return None


def put_bytes(namespace, key, payload):
    return _put(namespace, key, binary_payload=payload)


def get_bytes(namespace, key):
    engine, table = _initialize()
    if engine is None:
        return None
    try:
        from sqlalchemy import and_, select

        with engine.connect() as connection:
            return connection.execute(
                select(table.c.payload_binary).where(
                    and_(table.c.namespace == namespace, table.c.item_key == key)
                )
            ).scalar_one_or_none()
    except Exception:
        logger.exception("database binary read failed: namespace=%s key=%s", namespace, key)
        return None


def reset_for_tests():
    """Reset lazy state after changing DATABASE_URL in an isolated test."""
    global _engine, _table, _initialization_attempted, _last_initialization_failure
    if _engine is not None:
        _engine.dispose()
    _engine, _table, _initialization_attempted, _last_initialization_failure = None, None, False, 0.0
