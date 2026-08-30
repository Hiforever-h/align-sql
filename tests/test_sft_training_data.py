from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from align_sql.training.sft_data import prepare_split


class FakeChatTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        prompt_ids = [10, 11, 12, 13]
        if len(conversation) == 1:
            return prompt_ids if add_generation_prompt else prompt_ids[:-1]
        completion_length = int(conversation[1]["content"])
        return prompt_ids + [20] * completion_length


def _row(question_id: int, completion_length: int, recorded_length: int) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "db_id": "toy",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": str(completion_length)},
        ],
        "metadata": {"token_lengths": {"total": recorded_length}},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prepare_split_explicitly_drops_overlength_samples(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_row(1, 2, 6), _row(2, 7, 11)])

    dataset, audit = prepare_split(
        path,
        split="train",
        tokenizer=FakeChatTokenizer(),
        max_length=8,
        expected_dropped=1,
    )

    assert len(dataset) == 1
    assert dataset[0]["question_id"] == 1
    assert dataset[0]["prompt"] == [{"role": "user", "content": "question"}]
    assert audit.dropped_question_ids == (2,)
    assert audit.max_total_tokens == 6
    assert audit.min_completion_tokens == 2


def test_prepare_split_detects_tokenizer_length_drift(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [_row(1, 2, 999)])

    with pytest.raises(ValueError, match="Tokenizer length drift"):
        prepare_split(
            path,
            split="train",
            tokenizer=FakeChatTokenizer(),
            max_length=8,
            expected_dropped=0,
        )

