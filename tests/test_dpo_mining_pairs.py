from __future__ import annotations

import sqlite3
from pathlib import Path

import align_sql.training.dpo.pairs as pairs_module
from align_sql.training.dpo.mine import _completion_length_and_cap
from align_sql.training.dpo.pairs import (
    select_preference_pair,
    split_preference_pairs,
    verify_candidate_group,
)


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)")
    connection.executemany("INSERT INTO items VALUES (?, ?)", [(1, 10), (2, 20)])
    connection.commit()
    connection.close()
    return path


def _candidate(index: int, sql: str, tokens: int) -> dict[str, object]:
    return {
        "candidate_index": index,
        "generation": f"Reasoning\n```sql\n{sql}\n```",
        "input_tokens": 10,
        "generated_tokens": tokens,
        "hit_max_new_tokens": False,
        "batch_seed": 42,
        "generation_seconds": 1.0,
    }


def _raw_group() -> dict[str, object]:
    return {
        "source_index": 0,
        "question_id": 7,
        "db_id": "example",
        "prompt": [{"role": "user", "content": "return value"}],
        "gold_sql": "SELECT value FROM items WHERE id = 1",
        "candidates": [
            _candidate(0, "SELECT value FROM items WHERE id = 1", 50),
            _candidate(1, "SELECT value FROM items WHERE id = 2", 52),
            _candidate(2, "SELECT missing FROM items", 51),
            _candidate(3, "SELECT items.value FROM items WHERE items.id = 1", 51),
        ],
    }


def test_verification_executes_gold_once_and_prefers_executable_negative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = _database(tmp_path / "example.sqlite")
    calls: list[str] = []
    original = pairs_module.execute_read_only

    def counted_execute(*args, **kwargs):
        calls.append(str(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pairs_module, "execute_read_only", counted_execute)
    verified = verify_candidate_group(
        _raw_group(),
        database,
        timeout_seconds=1.0,
        max_result_rows=100,
    )

    assert calls.count("SELECT value FROM items WHERE id = 1") == 2
    # One call is the gold query and one is the identical correct candidate.
    assert len(calls) == 5
    pair, outcome = select_preference_pair(
        verified,
        include_execution_error_rejected=False,
    )
    assert outcome == "paired"
    assert pair is not None
    assert pair["metadata"]["negative_type"] == "execution_mismatch"
    assert pair["metadata"]["chosen_candidate_index"] == 0
    assert pair["metadata"]["rejected_candidate_index"] == 1
    assert pair["metadata"]["chosen_canonical_match"]
    assert len(pair["prompt"]) == 1


def test_invalid_gold_skips_candidate_execution(tmp_path: Path) -> None:
    database = _database(tmp_path / "example.sqlite")
    raw = _raw_group()
    raw["gold_sql"] = "SELECT value FROM missing_table"

    verified = verify_candidate_group(
        raw,
        database,
        timeout_seconds=1.0,
        max_result_rows=100,
    )

    assert verified["gold_execution"]["status"] == "sql_error"
    assert {
        candidate["execution"]["status"] for candidate in verified["candidates"]
    } == {"not_run_gold_invalid"}
    pair, outcome = select_preference_pair(
        verified,
        include_execution_error_rejected=False,
    )
    assert pair is None
    assert outcome == "gold_sql_error"


def test_pair_split_is_deterministic_and_exact_size() -> None:
    rows = [
        {
            "pair_id": f"pair-{index}",
            "question_id": index,
            "db_id": f"db-{index % 3}",
        }
        for index in range(20)
    ]

    first = split_preference_pairs(rows, validation_ratio=0.2, seed=42)
    second = split_preference_pairs(rows, validation_ratio=0.2, seed=42)

    assert first == second
    assert len(first[0]) == 16
    assert len(first[1]) == 4
    assert {row["pair_id"] for row in first[0]}.isdisjoint(
        row["pair_id"] for row in first[1]
    )


def test_completion_length_detects_eos_and_generation_cap() -> None:
    assert _completion_length_and_cap(
        [4, 5, 2, 2],
        eos_token_id=2,
        pad_token_id=2,
        max_new_tokens=4,
    ) == (3, False)
    assert _completion_length_and_cap(
        [4, 5, 6, 7],
        eos_token_id=2,
        pad_token_id=2,
        max_new_tokens=4,
    ) == (4, True)
