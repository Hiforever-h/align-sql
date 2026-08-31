from __future__ import annotations

import json
from pathlib import Path

from align_sql.training.dpo.data import load_mining_examples, select_mining_examples


def _write_rows(path: Path) -> None:
    rows = [
        {
            "question_id": index,
            "db_id": f"db_{index % 3}",
            "gold_sql": "SELECT 1",
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": "private supervised answer"},
            ],
        }
        for index in range(12)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_mining_selection_is_deterministic_and_does_not_leak_assistant(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    _write_rows(path)
    examples = load_mining_examples(path)

    first = select_mining_examples(examples, sample_size=6, seed=42)
    second = select_mining_examples(examples, sample_size=6, seed=42)

    assert first == second
    assert len(first) == 6
    assert {example.db_id for example in first} == {"db_0", "db_1", "db_2"}
    assert all(len(example.prompt) == 1 for example in first)
    assert all(example.prompt[0]["role"] == "user" for example in first)
    assert all("private supervised answer" not in example.prompt[0]["content"] for example in first)
