from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from datasets import Dataset


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class SplitAudit:
    split: str
    source_count: int
    kept_count: int
    dropped_count: int
    dropped_question_ids: tuple[int, ...]
    min_total_tokens: int
    max_total_tokens: int
    min_completion_tokens: int
    max_completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        return list(encoded["input_ids"])
    return list(encoded)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    if not rows:
        raise ValueError(f"SFT split is empty: {path}")
    return rows


def _validate_messages(row: dict[str, Any], path: Path) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"question_id={row.get('question_id')} in {path} must have two messages")
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant"]:
        raise ValueError(
            f"question_id={row.get('question_id')} in {path} has invalid roles: {roles}"
        )
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError(f"question_id={row.get('question_id')} in {path} has non-text content")
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def prepare_split(
    path: str | Path,
    *,
    split: str,
    tokenizer: ChatTokenizer,
    max_length: int,
    expected_dropped: int | None,
) -> tuple[Dataset, SplitAudit]:
    source_path = Path(path)
    rows = _read_jsonl(source_path)
    prepared: list[dict[str, Any]] = []
    dropped_ids: list[int] = []
    total_lengths: list[int] = []
    completion_lengths: list[int] = []

    for row in rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, int):
            raise ValueError(f"Invalid question_id in {source_path}: {question_id!r}")
        messages = _validate_messages(row, source_path)
        prompt = messages[:1]
        completion = messages[1:]
        full_ids = _input_ids(
            tokenizer.apply_chat_template(
                prompt + completion,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        prompt_ids = _input_ids(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "The tokenizer's prompt tokens are not a prefix of prompt+completion for "
                f"question_id={question_id}"
            )
        completion_length = len(full_ids) - len(prompt_ids)
        if completion_length <= 0:
            raise ValueError(f"No supervised completion tokens for question_id={question_id}")

        recorded_length = row.get("metadata", {}).get("token_lengths", {}).get("total")
        if recorded_length is not None and int(recorded_length) != len(full_ids):
            raise ValueError(
                f"Tokenizer length drift for question_id={question_id}: "
                f"stage1={recorded_length}, stage2={len(full_ids)}"
            )
        if len(full_ids) > max_length:
            dropped_ids.append(question_id)
            continue

        total_lengths.append(len(full_ids))
        completion_lengths.append(completion_length)
        prepared.append(
            {
                "prompt": prompt,
                "completion": completion,
                "question_id": question_id,
                "db_id": str(row["db_id"]),
            }
        )

    if expected_dropped is not None and len(dropped_ids) != expected_dropped:
        raise ValueError(
            f"Unexpected {split} overlength count: expected {expected_dropped}, "
            f"observed {len(dropped_ids)}"
        )
    if not prepared:
        raise ValueError(f"No {split} samples remain after max_length={max_length} filtering")

    audit = SplitAudit(
        split=split,
        source_count=len(rows),
        kept_count=len(prepared),
        dropped_count=len(dropped_ids),
        dropped_question_ids=tuple(dropped_ids),
        min_total_tokens=min(total_lengths),
        max_total_tokens=max(total_lengths),
        min_completion_tokens=min(completion_lengths),
        max_completion_tokens=max(completion_lengths),
    )
    return Dataset.from_list(prepared), audit
