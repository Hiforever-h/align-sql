"""Input loading and deterministic database-aware sampling for DPO mining."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MiningExample:
    source_index: int
    question_id: int
    db_id: str
    prompt: tuple[dict[str, str], ...]
    gold_sql: str


def _stable_score(seed: int, namespace: str, value: int | str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def load_mining_examples(path: str | Path) -> list[MiningExample]:
    source_path = Path(path)
    examples: list[MiningExample] = []
    seen_question_ids: set[int] = set()
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source_path}:{line_number}") from error
            question_id = row.get("question_id")
            db_id = row.get("db_id")
            gold_sql = row.get("gold_sql")
            messages = row.get("messages")
            if not isinstance(question_id, int) or question_id in seen_question_ids:
                raise ValueError(
                    f"Invalid or duplicate question_id at {source_path}:{line_number}"
                )
            if not isinstance(db_id, str) or not db_id:
                raise ValueError(f"Invalid db_id at {source_path}:{line_number}")
            if not isinstance(gold_sql, str) or not gold_sql.strip():
                raise ValueError(f"Invalid gold_sql at {source_path}:{line_number}")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(
                    f"Expected user and assistant messages at {source_path}:{line_number}"
                )
            user, assistant = messages
            if user.get("role") != "user" or assistant.get("role") != "assistant":
                raise ValueError(f"Invalid message roles at {source_path}:{line_number}")
            if not isinstance(user.get("content"), str) or not isinstance(
                assistant.get("content"), str
            ):
                raise ValueError(f"Non-text message at {source_path}:{line_number}")
            seen_question_ids.add(question_id)
            examples.append(
                MiningExample(
                    source_index=line_number - 1,
                    question_id=question_id,
                    db_id=db_id,
                    prompt=({"role": "user", "content": str(user["content"])},),
                    gold_sql=gold_sql,
                )
            )
    if not examples:
        raise ValueError(f"DPO mining data is empty: {source_path}")
    return examples


def select_mining_examples(
    examples: list[MiningExample],
    *,
    sample_size: int | None,
    seed: int,
) -> list[MiningExample]:
    """Select a deterministic subset while covering as many databases as possible."""

    if sample_size is None or sample_size >= len(examples):
        return sorted(
            examples,
            key=lambda item: _stable_score(seed, "question", item.question_id),
        )
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    by_database: dict[str, list[MiningExample]] = defaultdict(list)
    for example in examples:
        by_database[example.db_id].append(example)
    for db_id in by_database:
        by_database[db_id].sort(
            key=lambda item: _stable_score(seed, "question", item.question_id)
        )

    selected: list[MiningExample] = []
    selected_ids: set[int] = set()
    database_order = sorted(
        by_database,
        key=lambda db_id: _stable_score(seed, "database", db_id),
    )
    for db_id in database_order[:sample_size]:
        example = by_database[db_id][0]
        selected.append(example)
        selected_ids.add(example.question_id)

    if len(selected) < sample_size:
        remaining = sorted(
            (
                example
                for example in examples
                if example.question_id not in selected_ids
            ),
            key=lambda item: _stable_score(seed, "question", item.question_id),
        )
        selected.extend(remaining[: sample_size - len(selected)])

    return sorted(
        selected,
        key=lambda item: _stable_score(seed, "question", item.question_id),
    )


def selected_question_ids_sha256(examples: list[MiningExample]) -> str:
    payload = ",".join(str(example.question_id) for example in examples)
    return hashlib.sha256(payload.encode()).hexdigest()
