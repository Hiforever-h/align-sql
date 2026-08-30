from __future__ import annotations

import json
from pathlib import Path

from align_sql.evaluation.sft.data import load_eval_examples


def test_processed_eval_loader_does_not_leak_assistant_answer(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    row = {
        "question_id": 7,
        "db_id": "toy",
        "gold_sql": "SELECT secret FROM vault",
        "messages": [
            {"role": "user", "content": "show the value"},
            {"role": "assistant", "content": "DO_NOT_LEAK_THIS_ANSWER"},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    examples = load_eval_examples(path)

    assert len(examples) == 1
    assert examples[0].prompt == ({"role": "user", "content": "show the value"},)
    assert "DO_NOT_LEAK_THIS_ANSWER" not in str(examples[0].prompt)
    assert examples[0].gold_sql == "SELECT secret FROM vault"


def test_bird_eval_loader_supports_json_array_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "dev.json"
    rows = [
        {"input": "question one", "output": "SELECT 1", "db_id": "one"},
        {"input": "question two", "output": "SELECT 2", "db_id": "two"},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")

    examples = load_eval_examples(path, limit=1)

    assert len(examples) == 1
    assert examples[0].question_id == 0
    assert examples[0].prompt[0]["content"] == "question one"

