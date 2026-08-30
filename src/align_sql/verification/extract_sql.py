from __future__ import annotations

import re

import sqlglot
from sqlglot import ErrorLevel
from sqlglot.errors import SqlglotError

_FENCED_BLOCK = re.compile(
    r"```(?:\s*(?:sql|sqlite))?\s*\n?(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_SQL_START = re.compile(r"\b(?:SELECT|WITH)\b", flags=re.IGNORECASE)
_UNFENCED_SQL = re.compile(r"\b(?:SELECT|WITH)\b.*?;", flags=re.IGNORECASE | re.DOTALL)


def _trim_to_statement(text: str) -> str | None:
    start = _SQL_START.search(text)
    if start is None:
        return None
    statement = text[start.start() :].strip()
    semicolon = statement.find(";")
    if semicolon >= 0:
        statement = statement[: semicolon + 1]
    return statement.strip() or None


def extract_sql_candidates(response: str) -> list[str]:
    """Extract plausible SELECT/WITH statements, preferring Markdown code blocks."""
    candidates: list[str] = []
    for block in _FENCED_BLOCK.findall(response):
        statement = _trim_to_statement(block)
        if statement is not None:
            candidates.append(statement)

    if not candidates:
        for match in _UNFENCED_SQL.finditer(response):
            statement = _trim_to_statement(match.group(0))
            if statement is not None:
                candidates.append(statement)

    return list(dict.fromkeys(candidates))


def canonicalize_sql(sql: str) -> str | None:
    """Return a stable SQLite rendering for a single parseable SQL statement."""
    try:
        expressions = sqlglot.parse(sql.strip(), read="sqlite")
    except SqlglotError:
        return None
    if len(expressions) != 1 or expressions[0] is None:
        return None
    return expressions[0].sql(
        dialect="sqlite",
        pretty=False,
        normalize=True,
        unsupported_level=ErrorLevel.IGNORE,
    )


def sql_matches(candidate: str, gold: str) -> bool:
    candidate_canonical = canonicalize_sql(candidate)
    gold_canonical = canonicalize_sql(gold)
    if candidate_canonical is not None and gold_canonical is not None:
        return candidate_canonical == gold_canonical

    # Conservative fallback for dialect constructs not understood by SQLGlot.
    def normalize_text(value: str) -> str:
        return " ".join(value.rstrip("; ").split())

    return normalize_text(candidate) == normalize_text(gold)
