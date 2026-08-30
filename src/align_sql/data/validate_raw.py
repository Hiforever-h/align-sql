from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import ijson

from align_sql.data.config import SftDataConfig


def _load_train(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("BIRD train data must be a JSON array")
    return rows


def validate_raw(config: SftDataConfig) -> dict[str, Any]:
    train_rows = _load_train(config.train_path)
    counts: Counter[int] = Counter()
    invalid_shape = 0
    invalid_question_id = 0
    prompt_mismatches = 0
    non_monotonic_question_ids = 0
    previous_question_id = -1
    total = 0

    with config.synthetic_path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            total += 1
            question_id = item.get("question_id")
            messages = item.get("messages")
            if (
                not isinstance(question_id, int)
                or not isinstance(messages, list)
                or len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
                or not isinstance(messages[0].get("content"), str)
                or not isinstance(messages[1].get("content"), str)
            ):
                invalid_shape += 1
                continue
            if not 0 <= question_id < len(train_rows):
                invalid_question_id += 1
                continue
            if question_id < previous_question_id:
                non_monotonic_question_ids += 1
            previous_question_id = question_id
            counts[question_id] += 1
            if messages[0]["content"] != train_rows[question_id].get("input"):
                prompt_mismatches += 1

    count_histogram = Counter(counts.values())
    missing_ids = sorted(set(range(len(train_rows))) - set(counts))
    report = {
        "train_examples": len(train_rows),
        "synthetic_trajectories": total,
        "synthetic_question_ids": len(counts),
        "missing_question_ids": missing_ids,
        "missing_question_count": len(missing_ids),
        "trajectories_per_question_histogram": {
            str(key): value for key, value in sorted(count_histogram.items())
        },
        "invalid_shape_count": invalid_shape,
        "invalid_question_id_count": invalid_question_id,
        "prompt_mismatch_count": prompt_mismatches,
        "non_monotonic_question_id_count": non_monotonic_question_ids,
        "valid": all(
            value == 0
            for value in (
                invalid_shape,
                invalid_question_id,
                prompt_mismatches,
                non_monotonic_question_ids,
            )
        ),
    }
    config.raw_report_path.parent.mkdir(parents=True, exist_ok=True)
    with config.raw_report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate released BIRD SFT source data")
    parser.add_argument("--config", default="configs/data_sft.yaml")
    args = parser.parse_args()
    report = validate_raw(SftDataConfig.from_yaml(args.config))
    console_report = dict(report)
    console_report.pop("missing_question_ids")
    console_report["full_report_path"] = str(SftDataConfig.from_yaml(args.config).raw_report_path)
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit("Raw-data validation failed")


if __name__ == "__main__":
    main()
