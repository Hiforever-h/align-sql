"""Evaluation input loaders with explicit answer-leakage prevention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalExample:
    index: int
    question_id: int | str
    db_id: str
    prompt: tuple[dict[str, str], ...]
    gold_sql: str


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                rows.append(row)
        return rows

    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array of objects: {path}")
    return rows


def _from_processed(row: dict[str, Any], index: int) -> EvalExample:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"Processed example {index} must contain user and assistant messages")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError(f"Processed example {index} has invalid message roles")
    if not isinstance(user.get("content"), str) or not isinstance(
        assistant.get("content"), str
    ):
        raise ValueError(f"Processed example {index} has non-text messages")
    gold_sql = row.get("gold_sql")
    db_id = row.get("db_id")
    if not isinstance(gold_sql, str) or not isinstance(db_id, str):
        raise ValueError(f"Processed example {index} is missing gold_sql or db_id")
    question_id = row.get("question_id", index)
    if not isinstance(question_id, (int, str)):
        raise ValueError(f"Processed example {index} has invalid question_id")
    return EvalExample(
        index=index,
        question_id=question_id,
        db_id=db_id,
        prompt=({"role": "user", "content": str(user["content"])},),
        gold_sql=gold_sql,
    )


def _from_bird(row: dict[str, Any], index: int) -> EvalExample:
    prompt = row.get("input")
    gold_sql = row.get("output")
    db_id = row.get("db_id")
    if not isinstance(prompt, str) or not isinstance(gold_sql, str) or not isinstance(db_id, str):
        raise ValueError(f"BIRD example {index} is missing input, output, or db_id")
    question_id = row.get("question_id", index)
    if not isinstance(question_id, (int, str)):
        raise ValueError(f"BIRD example {index} has invalid question_id")
    return EvalExample(
        index=index,
        question_id=question_id,
        db_id=db_id,
        prompt=({"role": "user", "content": prompt},),
        gold_sql=gold_sql,
    )


def load_eval_examples(path: str | Path, limit: int | None = None) -> list[EvalExample]:
    source_path = Path(path)
    rows = _load_rows(source_path)
    if not rows:
        raise ValueError(f"Evaluation data is empty: {source_path}")
    examples: list[EvalExample] = []
    for index, row in enumerate(rows):
        if "messages" in row:
            example = _from_processed(row, index)
        elif {"input", "output", "db_id"}.issubset(row):
            example = _from_bird(row, index)
        else:
            raise ValueError(f"Unsupported evaluation row format at index {index}")
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break
    return examples

