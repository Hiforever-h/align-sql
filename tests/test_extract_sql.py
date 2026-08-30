from __future__ import annotations

from align_sql.verification.extract_sql import (
    canonicalize_sql,
    extract_sql_candidates,
    sql_matches,
)


def test_extract_fenced_sql() -> None:
    response = "Reasoning.\n```SQL\nSELECT name FROM users WHERE age > 30;\n```\nDone."
    assert extract_sql_candidates(response) == ["SELECT name FROM users WHERE age > 30;"]


def test_canonicalize_equivalent_formatting() -> None:
    left = "select name from users where age > 30;"
    right = "SELECT name\nFROM users\nWHERE age > 30"
    assert canonicalize_sql(left) == canonicalize_sql(right)
    assert sql_matches(left, right)


def test_reject_semantically_different_limit() -> None:
    candidate = "SELECT name FROM users ORDER BY score DESC;"
    gold = "SELECT name FROM users ORDER BY score DESC LIMIT 1;"
    assert not sql_matches(candidate, gold)


def test_tokenizer_error_uses_conservative_text_fallback() -> None:
    sql = r"SELECT title FROM movies WHERE tagline = 'An offer you can\'t refuse.';"
    assert canonicalize_sql(sql) is None
    assert sql_matches(sql, sql)
