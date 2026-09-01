"""Strict preference-data validation and token-length auditing for DPO."""

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
class PreferenceSplitAudit:
    split: str
    source_count: int
    database_count: int
    canonical_exact_chosen_count: int
    min_prompt_tokens: int
    max_prompt_tokens: int
    min_chosen_completion_tokens: int
    max_chosen_completion_tokens: int
    min_rejected_completion_tokens: int
    max_rejected_completion_tokens: int
    min_pair_max_tokens: int
    max_pair_max_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        values = encoded["input_ids"]
    else:
        values = encoded
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Expected a single tokenized conversation")
        values = values[0]
    return [int(value) for value in values]


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
        raise ValueError(f"DPO split is empty: {path}")
    return rows


def _messages(
    row: dict[str, Any],
    field: str,
    *,
    expected_role: str,
    path: Path,
) -> list[dict[str, str]]:
    messages = row.get(field)
    question_id = row.get("question_id")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError(
            f"question_id={question_id} in {path} must have one {field} message"
        )
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != expected_role:
        raise ValueError(
            f"question_id={question_id} in {path} has invalid {field} role"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"question_id={question_id} in {path} has empty {field} content"
        )
    return [{"role": expected_role, "content": content}]


def _validate_metadata(row: dict[str, Any], path: Path) -> bool:
    metadata = row.get("metadata")
    question_id = row.get("question_id")
    if not isinstance(metadata, dict):
        raise ValueError(f"question_id={question_id} in {path} has no metadata")
    if metadata.get("negative_type") != "execution_mismatch":
        raise ValueError(
            f"question_id={question_id} in {path} is not an execution-mismatch pair"
        )
    chosen_index = metadata.get("chosen_candidate_index")
    rejected_index = metadata.get("rejected_candidate_index")
    if not isinstance(chosen_index, int) or not isinstance(rejected_index, int):
        raise ValueError(f"question_id={question_id} in {path} has invalid candidate indexes")
    if chosen_index == rejected_index:
        raise ValueError(f"question_id={question_id} in {path} reuses one candidate")

    gold_digest = metadata.get("gold_digest")
    rejected_digest = metadata.get("rejected_digest")
    if not isinstance(gold_digest, str) or not isinstance(rejected_digest, str):
        raise ValueError(f"question_id={question_id} in {path} has invalid digests")
    if gold_digest == rejected_digest:
        raise ValueError(
            f"question_id={question_id} in {path} has an execution-matching rejected"
        )

    chosen_tokens = metadata.get("chosen_generated_tokens")
    rejected_tokens = metadata.get("rejected_generated_tokens")
    recorded_difference = metadata.get("token_length_difference")
    if not isinstance(chosen_tokens, int) or chosen_tokens <= 0:
        raise ValueError(f"question_id={question_id} in {path} has invalid chosen length")
    if not isinstance(rejected_tokens, int) or rejected_tokens <= 0:
        raise ValueError(f"question_id={question_id} in {path} has invalid rejected length")
    if recorded_difference != abs(chosen_tokens - rejected_tokens):
        raise ValueError(
            f"question_id={question_id} in {path} has inconsistent length metadata"
        )
    canonical = metadata.get("chosen_canonical_match")
    if not isinstance(canonical, bool):
        raise ValueError(
            f"question_id={question_id} in {path} has invalid canonical-match flag"
        )
    return canonical


def prepare_preference_split(
    path: str | Path,
    *,
    split: str,
    tokenizer: ChatTokenizer,
    max_length: int,
    expected_count: int | None,
) -> tuple[Dataset, PreferenceSplitAudit, set[int], set[str]]:
    source_path = Path(path)
    rows = _read_jsonl(source_path)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"Unexpected {split} pair count: expected {expected_count}, observed {len(rows)}"
        )

    prepared: list[dict[str, Any]] = []
    question_ids: set[int] = set()
    pair_ids: set[str] = set()
    databases: set[str] = set()
    canonical_count = 0
    prompt_lengths: list[int] = []
    chosen_completion_lengths: list[int] = []
    rejected_completion_lengths: list[int] = []
    pair_max_lengths: list[int] = []

    for row in rows:
        question_id = row.get("question_id")
        pair_id = row.get("pair_id")
        db_id = row.get("db_id")
        if not isinstance(question_id, int):
            raise ValueError(f"Invalid question_id in {source_path}: {question_id!r}")
        if question_id in question_ids:
            raise ValueError(f"Duplicate question_id={question_id} in {source_path}")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"question_id={question_id} in {source_path} has invalid pair_id")
        if pair_id in pair_ids:
            raise ValueError(f"Duplicate pair_id={pair_id} in {source_path}")
        if not isinstance(db_id, str) or not db_id:
            raise ValueError(f"question_id={question_id} in {source_path} has invalid db_id")

        prompt = _messages(row, "prompt", expected_role="user", path=source_path)
        chosen = _messages(row, "chosen", expected_role="assistant", path=source_path)
        rejected = _messages(row, "rejected", expected_role="assistant", path=source_path)
        if chosen[0]["content"] == rejected[0]["content"]:
            raise ValueError(
                f"question_id={question_id} in {source_path} has identical responses"
            )
        canonical_count += int(_validate_metadata(row, source_path))

        prompt_ids = _input_ids(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        chosen_ids = _input_ids(
            tokenizer.apply_chat_template(
                prompt + chosen,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        rejected_ids = _input_ids(
            tokenizer.apply_chat_template(
                prompt + rejected,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if chosen_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Tokenizer prompt prefix mismatch for chosen question_id={question_id}"
            )
        if rejected_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Tokenizer prompt prefix mismatch for rejected question_id={question_id}"
            )
        chosen_completion_length = len(chosen_ids) - len(prompt_ids)
        rejected_completion_length = len(rejected_ids) - len(prompt_ids)
        if chosen_completion_length <= 0 or rejected_completion_length <= 0:
            raise ValueError(f"No preference completion tokens for question_id={question_id}")
        pair_max_length = max(len(chosen_ids), len(rejected_ids))
        if pair_max_length > max_length:
            raise ValueError(
                f"DPO pair question_id={question_id} has {pair_max_length} tokens, "
                f"exceeding max_length={max_length}; truncation is forbidden"
            )

        question_ids.add(question_id)
        pair_ids.add(pair_id)
        databases.add(db_id)
        prompt_lengths.append(len(prompt_ids))
        chosen_completion_lengths.append(chosen_completion_length)
        rejected_completion_lengths.append(rejected_completion_length)
        pair_max_lengths.append(pair_max_length)
        prepared.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    audit = PreferenceSplitAudit(
        split=split,
        source_count=len(rows),
        database_count=len(databases),
        canonical_exact_chosen_count=canonical_count,
        min_prompt_tokens=min(prompt_lengths),
        max_prompt_tokens=max(prompt_lengths),
        min_chosen_completion_tokens=min(chosen_completion_lengths),
        max_chosen_completion_tokens=max(chosen_completion_lengths),
        min_rejected_completion_tokens=min(rejected_completion_lengths),
        max_rejected_completion_tokens=max(rejected_completion_lengths),
        min_pair_max_tokens=min(pair_max_lengths),
        max_pair_max_tokens=max(pair_max_lengths),
    )
    return Dataset.from_list(prepared), audit, question_ids, pair_ids


def prepare_preference_datasets(
    train_file: str | Path,
    validation_file: str | Path,
    *,
    tokenizer: ChatTokenizer,
    max_length: int,
    expected_train_pairs: int | None,
    expected_validation_pairs: int | None,
) -> tuple[Dataset, Dataset, PreferenceSplitAudit, PreferenceSplitAudit]:
    train_dataset, train_audit, train_questions, train_pairs = prepare_preference_split(
        train_file,
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        expected_count=expected_train_pairs,
    )
    validation_dataset, validation_audit, validation_questions, validation_pairs = (
        prepare_preference_split(
            validation_file,
            split="validation",
            tokenizer=tokenizer,
            max_length=max_length,
            expected_count=expected_validation_pairs,
        )
    )
    question_overlap = train_questions & validation_questions
    if question_overlap:
        raise ValueError(
            f"DPO train/validation question overlap: {sorted(question_overlap)}"
        )
    pair_overlap = train_pairs & validation_pairs
    if pair_overlap:
        raise ValueError(f"DPO train/validation pair overlap: {sorted(pair_overlap)}")
    return train_dataset, validation_dataset, train_audit, validation_audit


def validate_mining_manifest(
    path: str | Path,
    *,
    train_count: int,
    validation_count: int,
) -> dict[str, Any]:
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError(f"DPO mining manifest is not complete: {source_path}")
    report = manifest.get("mining_report")
    if not isinstance(report, dict):
        raise ValueError(f"DPO mining manifest has no mining_report: {source_path}")
    expected = {
        "train_pair_count": train_count,
        "validation_pair_count": validation_count,
        "pair_count": train_count + validation_count,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"Mining manifest {key}={report.get(key)!r}, expected {value}"
            )
    return manifest
