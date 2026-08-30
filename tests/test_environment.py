from __future__ import annotations

import sqlglot

import align_sql


def test_package_version() -> None:
    assert align_sql.__version__ == "0.1.0"


def test_sqlglot_can_parse_select() -> None:
    expression = sqlglot.parse_one("SELECT 1", read="sqlite")
    assert expression.sql(dialect="sqlite") == "SELECT 1"

