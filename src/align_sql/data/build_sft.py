from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from align_sql.data.config import SftDataConfig
from align_sql.verification.extract_sql import extract_sql_candidates, sql_matches


@dataclass(frozen=True)
class ValidTrajectory:
    question_id: int
    path_index: int
    response: str
    extracted_sql: str


def _load_train(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("BIRD train data must be a JSON array")
    return rows


def _choose_median_length(candidates: list[ValidTrajectory]) -> ValidTrajectory:
    ordered = sorted(candidates, key=lambda item: (len(item.response), item.path_index))
    return ordered[(len(ordered) - 1) // 2]


def _find_matching_sql(response: str, gold_sql: str) -> tuple[str | None, int]:
    candidates = extract_sql_candidates(response)
    for candidate in candidates:
        if sql_matches(candidate, gold_sql):
            return candidate, len(candidates)
    return None, len(candidates)


def select_trajectories(
    config: SftDataConfig,
    train_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    current_question_id: int | None = None
    current_candidates: list[ValidTrajectory] = []
    path_index = 0
    source_question_ids: set[int] = set()
    trajectories_total = 0
    trajectories_with_fence_candidates = 0
    trajectories_matching_gold = 0
    prompt_mismatches = 0
    invalid_rows = 0
    valid_paths_histogram: Counter[int] = Counter()

    def finalize_group() -> None:
        nonlocal current_candidates
        if current_question_id is None:
            return
        valid_paths_histogram[len(current_candidates)] += 1
        if not current_candidates:
            return
        chosen = _choose_median_length(current_candidates)
        train_row = train_rows[current_question_id]
        selected.append(
            {
                "question_id": current_question_id,
                "db_id": train_row["db_id"],
                "gold_sql": train_row["output"],
                "messages": [
                    {"role": "user", "content": train_row["input"]},
                    {"role": "assistant", "content": chosen.response},
                ],
                "metadata": {
                    "source_path_index": chosen.path_index,
                    "valid_paths_for_question": len(current_candidates),
                    "selection_method": config.selection_method,
                    "extracted_sql": chosen.extracted_sql,
                },
            }
        )
        current_candidates = []

    with config.synthetic_path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            trajectories_total += 1
            question_id = item.get("question_id")
            messages = item.get("messages")
            if (
                not isinstance(question_id, int)
                or not 0 <= question_id < len(train_rows)
                or not isinstance(messages, list)
                or len(messages) != 2
            ):
                invalid_rows += 1
                continue

            if current_question_id is None:
                current_question_id = question_id
            elif question_id != current_question_id:
                if question_id < current_question_id:
                    raise ValueError("Synthetic trajectories are not ordered by question_id")
                finalize_group()
                current_question_id = question_id
                path_index = 0

            source_question_ids.add(question_id)
            user_content = messages[0].get("content")
            response = messages[1].get("content")
            if user_content != train_rows[question_id].get("input"):
                prompt_mismatches += 1
                path_index += 1
                continue
            if not isinstance(response, str):
                invalid_rows += 1
                path_index += 1
                continue

            matching_sql, extracted_count = _find_matching_sql(
                response,
                train_rows[question_id]["output"],
            )
            if extracted_count:
                trajectories_with_fence_candidates += 1
            if matching_sql is not None:
                trajectories_matching_gold += 1
                current_candidates.append(
                    ValidTrajectory(
                        question_id=question_id,
                        path_index=path_index,
                        response=response,
                        extracted_sql=matching_sql,
                    )
                )
            path_index += 1

    finalize_group()
    selected_ids = {item["question_id"] for item in selected}
    no_gold_match_ids = sorted(source_question_ids - selected_ids)
    report = {
        "selection_policy": {
            "sql_requirement": "extracted SQL must match gold after SQLGlot canonicalization",
            "trajectory_choice": "median response length among gold-matching trajectories",
            "fallback": "none; questions without a gold-matching trajectory are excluded",
        },
        "source_trajectory_count": trajectories_total,
        "source_question_count": len(source_question_ids),
        "selected_question_count": len(selected),
        "questions_without_released_trajectory": len(train_rows) - len(source_question_ids),
        "questions_with_trajectories_but_no_gold_matching_sql": len(no_gold_match_ids),
        "question_ids_without_gold_matching_sql": no_gold_match_ids,
        "trajectories_with_extracted_sql": trajectories_with_fence_candidates,
        "trajectories_matching_gold_sql": trajectories_matching_gold,
        "prompt_mismatch_count": prompt_mismatches,
        "invalid_row_count": invalid_rows,
        "valid_paths_per_question_histogram": {
            str(key): value for key, value in sorted(valid_paths_histogram.items())
        },
        "selected_question_ids_sha256": hashlib.sha256(
            ",".join(str(value) for value in sorted(selected_ids)).encode()
        ).hexdigest(),
    }
    return selected, report


def _stable_score(seed: int, question_id: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def split_by_database(
    samples: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_database[sample["db_id"]].append(sample)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for db_id in sorted(by_database):
        database_samples = sorted(
            by_database[db_id],
            key=lambda item: _stable_score(seed, item["question_id"]),
        )
        validation_count = max(1, round(len(database_samples) * validation_ratio))
        validation.extend(database_samples[:validation_count])
        train.extend(database_samples[validation_count:])

    train.sort(key=lambda item: item["question_id"])
    validation.sort(key=lambda item: item["question_id"])
    return train, validation


def _percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def _encoded_length(encoded: Any) -> int:
    if isinstance(encoded, Mapping):
        return len(encoded["input_ids"])
    return len(encoded)


def add_token_lengths(
    samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    cutoff_candidates: Iterable[int],
    target_coverage: float,
) -> dict[str, Any]:
    total_lengths: list[int] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    for index, sample in enumerate(samples, start=1):
        messages = sample["messages"]
        total_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
        )
        response_tokens = tokenizer.encode(messages[1]["content"], add_special_tokens=False)
        lengths = {
            "total": _encoded_length(total_tokens),
            "prompt": _encoded_length(prompt_tokens),
            "response": len(response_tokens),
        }
        sample["metadata"]["token_lengths"] = lengths
        total_lengths.append(lengths["total"])
        prompt_lengths.append(lengths["prompt"])
        response_lengths.append(lengths["response"])
        if index % 1000 == 0:
            print(f"Tokenized {index}/{len(samples)} selected samples", flush=True)

    total_lengths.sort()
    prompt_lengths.sort()
    response_lengths.sort()
    coverage = {
        str(cutoff): sum(length <= cutoff for length in total_lengths) / len(total_lengths)
        for cutoff in cutoff_candidates
    }
    recommended_cutoff = max(cutoff_candidates)
    for cutoff in cutoff_candidates:
        if coverage[str(cutoff)] >= target_coverage:
            recommended_cutoff = cutoff
            break
    lossless_cutoff = next(
        (cutoff for cutoff in cutoff_candidates if coverage[str(cutoff)] == 1.0),
        None,
    )

    def summarize(values: list[int]) -> dict[str, int | float]:
        return {
            "min": values[0],
            "mean": round(sum(values) / len(values), 2),
            "p50": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": values[-1],
        }

    return {
        "tokenizer": tokenizer.name_or_path,
        "total_tokens": summarize(total_lengths),
        "prompt_tokens": summarize(prompt_lengths),
        "response_tokens": summarize(response_lengths),
        "cutoff_coverage": {key: round(value, 6) for key, value in coverage.items()},
        "target_coverage": target_coverage,
        "recommended_cutoff": recommended_cutoff,
        "lossless_cutoff": lossless_cutoff,
        "samples_over_recommended_cutoff": sum(
            length > recommended_cutoff for length in total_lengths
        ),
    }


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sft(config: SftDataConfig) -> dict[str, Any]:
    train_rows = _load_train(config.train_path)
    samples, report = select_trajectories(config, train_rows)
    if not samples:
        raise RuntimeError("No gold-matching synthetic trajectories were selected")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, use_fast=True)
    report["token_lengths"] = add_token_lengths(
        samples,
        tokenizer,
        config.cutoff_candidates,
        config.target_coverage,
    )
    train, validation = split_by_database(
        samples,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    report["split"] = {
        "method": "database-stratified deterministic hash",
        "seed": config.seed,
        "validation_ratio": config.validation_ratio,
        "train_count": len(train),
        "validation_count": len(validation),
        "database_count": len({item["db_id"] for item in samples}),
    }

    _write_jsonl(config.sft_train_path, train)
    _write_jsonl(config.sft_validation_path, validation)
    report["artifacts"] = {
        "train": {
            "path": str(config.sft_train_path),
            "count": len(train),
            "bytes": config.sft_train_path.stat().st_size,
            "sha256": _sha256_file(config.sft_train_path),
        },
        "validation": {
            "path": str(config.sft_validation_path),
            "count": len(validation),
            "bytes": config.sft_validation_path.stat().st_size,
            "sha256": _sha256_file(config.sft_validation_path),
        },
    }
    config.sft_report_path.parent.mkdir(parents=True, exist_ok=True)
    with config.sft_report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact, validated BIRD SFT dataset")
    parser.add_argument("--config", default="configs/data_sft.yaml")
    args = parser.parse_args()
    config = SftDataConfig.from_yaml(args.config)
    report = build_sft(config)
    console_report = dict(report)
    console_report.pop("question_ids_without_gold_matching_sql")
    console_report["full_report_path"] = str(config.sft_report_path)
    print(json.dumps(console_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
