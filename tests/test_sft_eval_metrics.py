from __future__ import annotations

from align_sql.evaluation.sft.metrics import analyze_generation, summarize_predictions


def test_analyze_generation_accepts_formatting_equivalence() -> None:
    analysis = analyze_generation(
        "Reasoning\n```sql\nselect name from users where age > 30;\n```",
        "SELECT name\nFROM users\nWHERE age > 30",
    )

    assert analysis.extraction_success is True
    assert analysis.parse_success is True
    assert analysis.canonical_match is True


def test_summarize_predictions_includes_execution_failures() -> None:
    records = [
        {
            "generation": "```sql\nSELECT 1;\n```",
            "generated_tokens": 5,
            "extraction_success": True,
            "parse_success": True,
            "canonical_match": True,
            "normalized_match": True,
            "execution": {
                "match": True,
                "candidate": {"status": "ok"},
            },
        },
        {
            "generation": "no sql",
            "generated_tokens": 2,
            "extraction_success": False,
            "parse_success": False,
            "canonical_match": False,
            "normalized_match": False,
            "execution": {
                "match": False,
                "candidate": {"status": "missing_sql"},
            },
        },
    ]

    summary = summarize_predictions(records)

    assert summary["sql_extraction_rate"] == 0.5
    assert summary["execution"]["accuracy"] == 0.5
    assert summary["execution"]["candidate_error_counts"] == {"missing_sql": 1}

