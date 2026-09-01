from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from align_sql.training.dpo.train_data import (
    prepare_preference_datasets,
    prepare_preference_split,
    validate_mining_manifest,
)


class FakeChatTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        assert tokenize is True
        prompt_ids = [10, 11, 12, 13]
        if len(conversation) == 1:
            assert add_generation_prompt is True
            return {"input_ids": prompt_ids}
        assert add_generation_prompt is False
        completion_length = int(conversation[1]["content"])
        return {"input_ids": prompt_ids + [20] * completion_length}


def _pair(
    question_id: int,
    *,
    chosen_length: int = 2,
    rejected_length: int = 3,
) -> dict[str, Any]:
    return {
        "pair_id": f"pair-{question_id}",
        "question_id": question_id,
        "db_id": f"db-{question_id % 2}",
        "prompt": [{"role": "user", "content": "question"}],
        "chosen": [{"role": "assistant", "content": str(chosen_length)}],
        "rejected": [{"role": "assistant", "content": str(rejected_length)}],
        "metadata": {
            "negative_type": "execution_mismatch",
            "chosen_candidate_index": 0,
            "rejected_candidate_index": 1,
            "chosen_generated_tokens": chosen_length,
            "rejected_generated_tokens": rejected_length,
            "token_length_difference": abs(chosen_length - rejected_length),
            "chosen_canonical_match": question_id % 2 == 0,
            "gold_digest": f"gold-{question_id}",
            "chosen_digest": f"gold-{question_id}",
            "rejected_digest": f"wrong-{question_id}",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prepare_preference_datasets_audits_lengths_and_leakage(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    _write_jsonl(train_file, [_pair(1), _pair(2, chosen_length=4, rejected_length=2)])
    _write_jsonl(validation_file, [_pair(3)])

    train, validation, train_audit, validation_audit = prepare_preference_datasets(
        train_file,
        validation_file,
        tokenizer=FakeChatTokenizer(),
        max_length=8,
        expected_train_pairs=2,
        expected_validation_pairs=1,
    )

    assert len(train) == 2
    assert len(validation) == 1
    assert set(train.column_names) == {"prompt", "chosen", "rejected"}
    assert train_audit.max_pair_max_tokens == 8
    assert train_audit.canonical_exact_chosen_count == 1
    assert validation_audit.database_count == 1


def test_prepare_preference_split_rejects_overlength_pair(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_pair(1, chosen_length=5, rejected_length=3)])

    with pytest.raises(ValueError, match="truncation is forbidden"):
        prepare_preference_split(
            path,
            split="train",
            tokenizer=FakeChatTokenizer(),
            max_length=8,
            expected_count=1,
        )


def test_prepare_preference_datasets_rejects_question_overlap(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    _write_jsonl(train_file, [_pair(1)])
    _write_jsonl(validation_file, [_pair(1)])

    with pytest.raises(ValueError, match="question overlap"):
        prepare_preference_datasets(
            train_file,
            validation_file,
            tokenizer=FakeChatTokenizer(),
            max_length=8,
            expected_train_pairs=1,
            expected_validation_pairs=1,
        )


def test_validate_mining_manifest_requires_matching_counts(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "mining_report": {
                    "pair_count": 3,
                    "train_pair_count": 2,
                    "validation_pair_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = validate_mining_manifest(path, train_count=2, validation_count=1)
    assert manifest["status"] == "complete"

    with pytest.raises(ValueError, match="train_pair_count"):
        validate_mining_manifest(path, train_count=1, validation_count=1)
