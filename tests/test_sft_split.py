from __future__ import annotations

from align_sql.data.build_sft import _encoded_length, split_by_database


def test_split_is_deterministic_and_keeps_each_database_in_validation() -> None:
    samples = [
        {"question_id": index, "db_id": "a" if index < 10 else "b"}
        for index in range(20)
    ]
    first_train, first_validation = split_by_database(samples, 0.2, 42)
    second_train, second_validation = split_by_database(samples, 0.2, 42)

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 16
    assert len(first_validation) == 4
    assert {sample["db_id"] for sample in first_validation} == {"a", "b"}


def test_encoded_length_supports_transformers_batch_encoding_shape() -> None:
    assert _encoded_length({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3
    assert _encoded_length([1, 2]) == 2
