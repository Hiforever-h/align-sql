"""Inline execution verification and preference-pair construction."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from align_sql.evaluation.sft.execution import (
    compare_query_results,
    execute_read_only,
)
from align_sql.evaluation.sft.metrics import analyze_generation


def _not_run(status: str, error: str) -> dict[str, Any]:
    return {
        "status": status,
        "row_count": 0,
        "digest": None,
        "elapsed_seconds": 0.0,
        "error": error,
    }


def verify_candidate_group(
    raw_group: dict[str, Any],
    database_path: str | Path,
    *,
    timeout_seconds: float,
    max_result_rows: int,
) -> dict[str, Any]:
    """Execute gold once inline, then verify every candidate against the cached result."""

    gold_sql = raw_group.get("gold_sql")
    candidates = raw_group.get("candidates")
    if not isinstance(gold_sql, str) or not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate group must contain gold_sql and non-empty candidates")

    gold = execute_read_only(
        database_path,
        gold_sql,
        timeout_seconds=timeout_seconds,
        max_result_rows=max_result_rows,
    )
    verified_candidates: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        generation = raw_candidate.get("generation")
        if not isinstance(generation, str):
            raise ValueError("Candidate generation must be text")
        analysis = analyze_generation(generation, gold_sql)
        matched = False
        order_sensitive: bool | None = None
        if gold.status != "ok":
            candidate_execution = _not_run(
                "not_run_gold_invalid",
                f"Gold execution status is {gold.status}",
            )
        elif analysis.extracted_sql is None:
            candidate_execution = _not_run(
                "missing_sql",
                "No SQL could be extracted from the generation",
            )
        else:
            candidate_result = execute_read_only(
                database_path,
                analysis.extracted_sql,
                timeout_seconds=timeout_seconds,
                max_result_rows=max_result_rows,
            )
            matched, order_sensitive = compare_query_results(
                candidate_result,
                gold,
                gold_sql,
            )
            candidate_execution = candidate_result.public()
        verified_candidates.append(
            {
                **raw_candidate,
                **analysis.to_dict(),
                "execution": candidate_execution,
                "execution_match": matched,
                "order_sensitive": order_sensitive,
            }
        )

    return {
        **{key: value for key, value in raw_group.items() if key != "candidates"},
        "gold_execution": gold.public(),
        "candidates": verified_candidates,
    }


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_generations: set[str] = set()
    seen_canonical_sql: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: int(item["candidate_index"])):
        generation = str(candidate["generation"]).strip()
        canonical_sql = candidate.get("canonical_sql")
        if generation in seen_generations:
            continue
        if isinstance(canonical_sql, str) and canonical_sql in seen_canonical_sql:
            continue
        seen_generations.add(generation)
        if isinstance(canonical_sql, str):
            seen_canonical_sql.add(canonical_sql)
        unique.append(candidate)
    return unique


def select_preference_pair(
    verified_group: dict[str, Any],
    *,
    include_execution_error_rejected: bool,
) -> tuple[dict[str, Any] | None, str]:
    """Choose one length-balanced correct-vs-hard-negative pair for a prompt."""

    gold_execution = verified_group.get("gold_execution", {})
    if gold_execution.get("status") != "ok":
        return None, f"gold_{gold_execution.get('status', 'missing')}"

    candidates = _deduplicate_candidates(list(verified_group["candidates"]))
    eligible = [
        candidate
        for candidate in candidates
        if not bool(candidate.get("hit_max_new_tokens"))
    ]
    chosen_pool = [candidate for candidate in eligible if candidate["execution_match"]]
    hard_negative_pool = [
        candidate
        for candidate in eligible
        if candidate["execution"]["status"] == "ok"
        and not candidate["execution_match"]
    ]
    error_negative_pool = [
        candidate
        for candidate in eligible
        if candidate["execution"]["status"]
        in {"rejected", "sql_error", "database_error", "timeout"}
    ]
    if not chosen_pool:
        return None, "no_correct_candidate"
    if hard_negative_pool:
        rejected_pool = hard_negative_pool
        negative_type = "execution_mismatch"
    elif include_execution_error_rejected and error_negative_pool:
        rejected_pool = error_negative_pool
        negative_type = "execution_error"
    else:
        return None, "no_hard_negative"

    chosen, rejected = min(
        (
            (chosen_candidate, rejected_candidate)
            for chosen_candidate in chosen_pool
            for rejected_candidate in rejected_pool
        ),
        key=lambda pair: (
            abs(
                int(pair[0]["generated_tokens"])
                - int(pair[1]["generated_tokens"])
            ),
            not bool(pair[0]["canonical_match"]),
            int(pair[0]["candidate_index"]),
            int(pair[1]["candidate_index"]),
        ),
    )
    if chosen.get("canonical_sql") == rejected.get("canonical_sql"):
        raise ValueError("Selected chosen and rejected candidates have identical SQL")

    pair_id_payload = (
        f"{verified_group['question_id']}\n{chosen['generation']}\n{rejected['generation']}"
    )
    pair = {
        "pair_id": hashlib.sha256(pair_id_payload.encode()).hexdigest(),
        "question_id": verified_group["question_id"],
        "db_id": verified_group["db_id"],
        "prompt": verified_group["prompt"],
        "chosen": [{"role": "assistant", "content": chosen["generation"]}],
        "rejected": [{"role": "assistant", "content": rejected["generation"]}],
        "metadata": {
            "negative_type": negative_type,
            "chosen_candidate_index": chosen["candidate_index"],
            "rejected_candidate_index": rejected["candidate_index"],
            "chosen_sql": chosen["extracted_sql"],
            "rejected_sql": rejected["extracted_sql"],
            "chosen_generated_tokens": chosen["generated_tokens"],
            "rejected_generated_tokens": rejected["generated_tokens"],
            "token_length_difference": abs(
                int(chosen["generated_tokens"])
                - int(rejected["generated_tokens"])
            ),
            "chosen_canonical_match": chosen["canonical_match"],
            "gold_digest": gold_execution["digest"],
            "chosen_digest": chosen["execution"]["digest"],
            "rejected_digest": rejected["execution"]["digest"],
        },
    }
    return pair, "paired"


def _stable_pair_score(seed: int, namespace: str, pair: dict[str, Any]) -> str:
    payload = f"{seed}:{namespace}:{pair['db_id']}:{pair['question_id']}"
    return hashlib.sha256(payload.encode()).hexdigest()


def split_preference_pairs(
    pairs: list[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(pairs) < 2:
        raise ValueError("At least two preference pairs are required for a split")
    target_validation = max(1, min(len(pairs) - 1, round(len(pairs) * validation_ratio)))
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_database[str(pair["db_id"])].append(pair)
    for db_id in by_database:
        by_database[db_id].sort(
            key=lambda pair: _stable_pair_score(seed, "pair", pair)
        )

    validation: list[dict[str, Any]] = []
    validation_ids: set[str] = set()
    eligible_databases = sorted(
        (db_id for db_id, rows in by_database.items() if len(rows) >= 2),
        key=lambda db_id: hashlib.sha256(f"{seed}:database:{db_id}".encode()).hexdigest(),
    )
    for db_id in eligible_databases[:target_validation]:
        pair = by_database[db_id][0]
        validation.append(pair)
        validation_ids.add(str(pair["pair_id"]))

    if len(validation) < target_validation:
        remaining = sorted(
            (pair for pair in pairs if str(pair["pair_id"]) not in validation_ids),
            key=lambda pair: _stable_pair_score(seed, "remaining", pair),
        )
        validation.extend(remaining[: target_validation - len(validation)])
        validation_ids.update(str(pair["pair_id"]) for pair in validation)

    train = [pair for pair in pairs if str(pair["pair_id"]) not in validation_ids]
    train.sort(key=lambda pair: (str(pair["db_id"]), int(pair["question_id"])))
    validation.sort(key=lambda pair: (str(pair["db_id"]), int(pair["question_id"])))
    return train, validation


def summarize_verified_groups(
    groups: list[dict[str, Any]],
    pair_outcomes: Counter[str],
    pairs: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
    validation_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [candidate for group in groups for candidate in group["candidates"]]
    candidate_statuses = Counter(
        str(candidate["execution"]["status"]) for candidate in candidates
    )
    gold_statuses = Counter(str(group["gold_execution"]["status"]) for group in groups)
    negative_types = Counter(str(pair["metadata"]["negative_type"]) for pair in pairs)
    return {
        "questions": len(groups),
        "candidates": len(candidates),
        "gold_statuses": dict(sorted(gold_statuses.items())),
        "sql_extraction_count": sum(
            bool(candidate["extraction_success"]) for candidate in candidates
        ),
        "sql_parse_count": sum(bool(candidate["parse_success"]) for candidate in candidates),
        "candidate_execution_statuses": dict(sorted(candidate_statuses.items())),
        "execution_match_count": sum(
            bool(candidate["execution_match"]) for candidate in candidates
        ),
        "pair_outcomes": dict(sorted(pair_outcomes.items())),
        "pair_count": len(pairs),
        "pair_yield": round(len(pairs) / len(groups), 6) if groups else 0.0,
        "negative_types": dict(sorted(negative_types.items())),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "train_database_count": len({pair["db_id"] for pair in train_pairs}),
        "validation_database_count": len(
            {pair["db_id"] for pair in validation_pairs}
        ),
        "token_length_difference": _numeric_summary(
            [int(pair["metadata"]["token_length_difference"]) for pair in pairs]
        ),
    }


def _numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }
