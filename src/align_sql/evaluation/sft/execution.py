"""Read-only SQLite execution comparison for Text-to-SQL diagnostics."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import SqlglotError

_DENIED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ANALYZE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_UPDATE,
}


@dataclass(frozen=True)
class QueryResult:
    status: str
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    digest: str | None
    elapsed_seconds: float
    error: str | None

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "row_count": self.row_count,
            "digest": self.digest,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "error": self.error,
        }


def resolve_database(db_root: str | Path, db_id: str) -> Path:
    root = Path(db_root)
    candidates = (
        root / db_id / f"{db_id}.sqlite",
        root / db_id / f"{db_id}.db",
        root / f"{db_id}.sqlite",
        root / f"{db_id}.db",
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No SQLite database found for db_id={db_id!r} under {root}"
        )
    if len(existing) > 1:
        paths = ", ".join(str(path) for path in existing)
        raise ValueError(f"Ambiguous database files for db_id={db_id!r}: {paths}")
    return existing[0]


def _validate_read_only_query(sql: str) -> str | None:
    try:
        expressions = sqlglot.parse(sql.strip(), read="sqlite")
    except SqlglotError as error:
        return f"parse_error: {error}"
    if len(expressions) != 1 or not isinstance(expressions[0], exp.Query):
        return "Only one SELECT/WITH query is allowed"
    return None


def _normalize_value(value: Any) -> Any:
    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "inf" if value > 0 else "-inf"]
        return ["float", round(value, 12)]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, str):
        return ["str", value]
    return [type(value).__name__, str(value)]


def _normalize_rows(rows: list[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(_normalize_value(value) for value in row) for row in rows)


def _rows_digest(rows: tuple[tuple[Any, ...], ...]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _authorizer(
    action: int,
    parameter1: str | None,
    parameter2: str | None,
    database_name: str | None,
    trigger_name: str | None,
) -> int:
    del database_name, trigger_name
    if action in _DENIED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (parameter2 or parameter1) == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def execute_read_only(
    database_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float,
    max_result_rows: int,
) -> QueryResult:
    started = time.monotonic()
    validation_error = _validate_read_only_query(sql)
    if validation_error is not None:
        return QueryResult(
            status="rejected",
            rows=(),
            row_count=0,
            digest=None,
            elapsed_seconds=time.monotonic() - started,
            error=validation_error,
        )

    database_uri = f"{Path(database_path).resolve().as_uri()}?mode=ro"
    deadline = started + timeout_seconds
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_uri, uri=True, timeout=timeout_seconds)
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_authorizer)
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            10_000,
        )
        cursor = connection.execute(sql)
        rows: list[tuple[Any, ...]] = []
        while True:
            batch = cursor.fetchmany(1_000)
            if not batch:
                break
            rows.extend(batch)
            if len(rows) > max_result_rows:
                return QueryResult(
                    status="too_many_rows",
                    rows=(),
                    row_count=len(rows),
                    digest=None,
                    elapsed_seconds=time.monotonic() - started,
                    error=f"Query exceeded max_result_rows={max_result_rows}",
                )
        normalized_rows = _normalize_rows(rows)
        return QueryResult(
            status="ok",
            rows=normalized_rows,
            row_count=len(normalized_rows),
            digest=_rows_digest(normalized_rows),
            elapsed_seconds=time.monotonic() - started,
            error=None,
        )
    except sqlite3.OperationalError as error:
        message = str(error)
        status = "timeout" if "interrupted" in message.lower() else "sql_error"
        return QueryResult(
            status=status,
            rows=(),
            row_count=0,
            digest=None,
            elapsed_seconds=time.monotonic() - started,
            error=message,
        )
    except sqlite3.DatabaseError as error:
        return QueryResult(
            status="database_error",
            rows=(),
            row_count=0,
            digest=None,
            elapsed_seconds=time.monotonic() - started,
            error=str(error),
        )
    finally:
        if connection is not None:
            connection.close()


def _has_top_level_order_by(sql: str) -> bool:
    try:
        expressions = sqlglot.parse(sql.strip(), read="sqlite")
    except SqlglotError:
        return True
    if len(expressions) != 1 or expressions[0] is None:
        return True
    return expressions[0].args.get("order") is not None


def _unordered_rows(rows: tuple[tuple[Any, ...], ...]) -> list[str]:
    return sorted(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
    )


def compare_execution(
    database_path: str | Path,
    candidate_sql: str,
    gold_sql: str,
    *,
    timeout_seconds: float,
    max_result_rows: int,
) -> dict[str, Any]:
    gold = execute_read_only(
        database_path,
        gold_sql,
        timeout_seconds=timeout_seconds,
        max_result_rows=max_result_rows,
    )
    candidate = execute_read_only(
        database_path,
        candidate_sql,
        timeout_seconds=timeout_seconds,
        max_result_rows=max_result_rows,
    )
    order_sensitive = _has_top_level_order_by(gold_sql)
    matched = False
    if gold.status == "ok" and candidate.status == "ok":
        matched = (
            candidate.rows == gold.rows
            if order_sensitive
            else _unordered_rows(candidate.rows) == _unordered_rows(gold.rows)
        )
    return {
        "match": matched,
        "order_sensitive": order_sensitive,
        "candidate": candidate.public(),
        "gold": gold.public(),
    }
