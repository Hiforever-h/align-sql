"""SQL extraction and aggregate metrics for SFT evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from align_sql.verification.extract_sql import (
    canonicalize_sql,
    extract_sql_candidates,
    sql_matches,
)


@dataclass(frozen=True)
class SqlAnalysis:
    extracted_sql: str | None
    extraction_success: bool
    parse_success: bool
    canonical_sql: str | None
    canonical_gold_sql: str | None
    canonical_match: bool
    normalized_match: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_generation(generation: str, gold_sql: str) -> SqlAnalysis:
    candidates = extract_sql_candidates(generation)
    extracted_sql = candidates[0] if candidates else None
    canonical_sql = canonicalize_sql(extracted_sql) if extracted_sql is not None else None
    canonical_gold = canonicalize_sql(gold_sql)
    return SqlAnalysis(
        extracted_sql=extracted_sql,
        extraction_success=extracted_sql is not None,
        parse_success=canonical_sql is not None,
        canonical_sql=canonical_sql,
        canonical_gold_sql=canonical_gold,
        canonical_match=(
            canonical_sql is not None
            and canonical_gold is not None
            and canonical_sql == canonical_gold
        ),
        normalized_match=(
            extracted_sql is not None and sql_matches(extracted_sql, gold_sql)
        ),
    )


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty prediction set")
    total = len(records)

    def count_flag(name: str) -> int:
        return sum(bool(record[name]) for record in records)

    def rate(count: int) -> float:
        return round(count / total, 6)

    extraction_count = count_flag("extraction_success")
    parse_count = count_flag("parse_success")
    canonical_count = count_flag("canonical_match")
    normalized_count = count_flag("normalized_match")
    nonempty_count = sum(bool(record["generation"].strip()) for record in records)
    generated_tokens = [int(record["generated_tokens"]) for record in records]
    summary: dict[str, Any] = {
        "total": total,
        "nonempty_generation_count": nonempty_count,
        "nonempty_generation_rate": rate(nonempty_count),
        "sql_extraction_count": extraction_count,
        "sql_extraction_rate": rate(extraction_count),
        "sql_parse_count": parse_count,
        "sql_parse_rate": rate(parse_count),
        "canonical_match_count": canonical_count,
        "canonical_match_accuracy": rate(canonical_count),
        "normalized_match_count": normalized_count,
        "normalized_match_accuracy": rate(normalized_count),
        "generated_tokens": {
            "min": min(generated_tokens),
            "max": max(generated_tokens),
            "mean": round(sum(generated_tokens) / total, 2),
        },
    }

    execution_records = [record.get("execution") for record in records]
    if any(value is not None for value in execution_records):
        execution_matches = sum(
            bool(value and value.get("match")) for value in execution_records
        )
        candidate_success = sum(
            bool(value and value.get("candidate", {}).get("status") == "ok")
            for value in execution_records
        )
        error_counts = Counter(
            str(value.get("candidate", {}).get("status"))
            for value in execution_records
            if value and value.get("candidate", {}).get("status") != "ok"
        )
        summary["execution"] = {
            "match_count": execution_matches,
            "accuracy": rate(execution_matches),
            "candidate_success_count": candidate_success,
            "candidate_success_rate": rate(candidate_success),
            "candidate_error_counts": dict(sorted(error_counts.items())),
        }
    return summary

