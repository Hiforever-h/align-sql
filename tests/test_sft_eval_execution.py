from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from align_sql.evaluation.sft.execution import (
    compare_execution,
    execute_read_only,
    resolve_database,
)


@pytest.fixture
def toy_database(tmp_path: Path) -> tuple[Path, Path]:
    db_root = tmp_path / "databases"
    database_dir = db_root / "toy"
    database_dir.mkdir(parents=True)
    database_path = database_dir / "toy.sqlite"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, score REAL);
        INSERT INTO items (name, score) VALUES ('beta', 20.0);
        INSERT INTO items (name, score) VALUES ('alpha', 30.0);
        INSERT INTO items (name, score) VALUES ('gamma', 10.0);
        """
    )
    connection.close()
    return db_root, database_path


def test_resolve_bird_database_layout(toy_database: tuple[Path, Path]) -> None:
    db_root, database_path = toy_database
    assert resolve_database(db_root, "toy") == database_path


def test_execution_matches_equivalent_unordered_results(
    toy_database: tuple[Path, Path],
) -> None:
    _, database_path = toy_database
    comparison = compare_execution(
        database_path,
        "SELECT name FROM items WHERE score > 10 ORDER BY name DESC",
        "SELECT name FROM items WHERE score >= 20",
        timeout_seconds=1.0,
        max_result_rows=100,
    )

    assert comparison["order_sensitive"] is False
    assert comparison["match"] is True


def test_execution_respects_gold_order_by(toy_database: tuple[Path, Path]) -> None:
    _, database_path = toy_database
    comparison = compare_execution(
        database_path,
        "SELECT name FROM items ORDER BY name DESC",
        "SELECT name FROM items ORDER BY name ASC",
        timeout_seconds=1.0,
        max_result_rows=100,
    )

    assert comparison["order_sensitive"] is True
    assert comparison["match"] is False


def test_execution_rejects_mutation(toy_database: tuple[Path, Path]) -> None:
    _, database_path = toy_database
    result = execute_read_only(
        database_path,
        "DELETE FROM items",
        timeout_seconds=1.0,
        max_result_rows=100,
    )

    assert result.status == "rejected"
    connection = sqlite3.connect(database_path)
    assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3
    connection.close()


def test_execution_enforces_result_limit(toy_database: tuple[Path, Path]) -> None:
    _, database_path = toy_database
    result = execute_read_only(
        database_path,
        "SELECT * FROM items",
        timeout_seconds=1.0,
        max_result_rows=2,
    )

    assert result.status == "too_many_rows"


def test_execution_timeout_interrupts_query(toy_database: tuple[Path, Path]) -> None:
    _, database_path = toy_database
    result = execute_read_only(
        database_path,
        """
        WITH RECURSIVE counter(value) AS (
          SELECT 1
          UNION ALL
          SELECT value + 1 FROM counter WHERE value < 1000000000
        )
        SELECT SUM(value) FROM counter;
        """,
        timeout_seconds=0.001,
        max_result_rows=100,
    )

    assert result.status == "timeout"

