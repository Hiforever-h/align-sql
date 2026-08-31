"""Generate, verify, and pair SFT-policy trajectories for DPO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from align_sql.evaluation.sft.execution import resolve_database
from align_sql.training.dpo.config import DpoMiningConfig
from align_sql.training.dpo.data import (
    MiningExample,
    load_mining_examples,
    select_mining_examples,
    selected_question_ids_sha256,
)
from align_sql.training.dpo.pairs import (
    select_preference_pair,
    split_preference_pairs,
    summarize_verified_groups,
    verify_candidate_group,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine execution-guided DPO preferences from an SFT adapter"
    )
    parser.add_argument("--config", default="configs/dpo_mining.yaml")
    parser.add_argument("--stage", choices=("generate", "build", "all"), default="all")
    parser.add_argument("--adapter-path")
    parser.add_argument("--db-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing question groups and rebuild derived pair files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts owned by the selected stage",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and deterministic prompt selection without CUDA or databases",
    )
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch(examples: list[MiningExample], batch_size: int) -> Iterator[list[MiningExample]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


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
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _selection_fingerprint(
    config: DpoMiningConfig,
    selected: list[MiningExample],
) -> dict[str, Any]:
    return {
        "data_sha256": _sha256_file(config.data.path),
        "selected_question_ids_sha256": selected_question_ids_sha256(selected),
        "selected_questions": len(selected),
        "num_candidates": config.sampling.num_candidates,
        "adapter_path": str(config.model.adapter_path),
        "sampling_seed": config.sampling.seed,
    }


def _new_manifest(
    config: DpoMiningConfig,
    selected: list[MiningExample],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config.source_path),
        "status": "initialized",
        "fingerprint": _selection_fingerprint(config, selected),
        "model": {
            **asdict(config.model),
            "adapter_path": str(config.model.adapter_path),
        },
        "data": {**asdict(config.data), "path": str(config.data.path)},
        "sampling": asdict(config.sampling),
        "execution": {
            **asdict(config.execution),
            "db_root": str(config.execution.db_root),
            "gold_execution_policy": "inline_once_per_question_no_prevalidation_pass",
        },
        "pairing": asdict(config.pairing),
        "output_dir": str(config.output.directory),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            **{
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "bitsandbytes",
                    "sqlglot",
                )
            },
        },
        "hf_cache": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        },
    }


def _prepare_generation_artifacts(
    config: DpoMiningConfig,
    selected: list[MiningExample],
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    candidates_file = config.output.candidates_file
    manifest_file = config.output.manifest_file
    if overwrite:
        candidates_file.parent.mkdir(parents=True, exist_ok=True)
        candidates_file.write_text("", encoding="utf-8")
        manifest = _new_manifest(config, selected)
        _write_json(manifest_file, manifest)
        return {}, manifest
    if resume:
        if not candidates_file.is_file() or not manifest_file.is_file():
            raise FileNotFoundError(
                "--resume requires existing candidates.jsonl and run_manifest.json"
            )
        manifest = _read_json(manifest_file)
        expected = _selection_fingerprint(config, selected)
        if manifest.get("fingerprint") != expected:
            raise ValueError("Resume fingerprint does not match the current mining config")
        rows = _read_jsonl(candidates_file)
        existing = {int(row["question_id"]): row for row in rows}
        if len(existing) != len(rows):
            raise ValueError("Duplicate question groups in candidates.jsonl")
        return existing, manifest
    if candidates_file.exists() or manifest_file.exists():
        raise FileExistsError(
            f"Generation artifacts already exist under {config.output.directory}; "
            "use --resume, --overwrite, or a new --output-dir"
        )
    config.output.directory.mkdir(parents=True, exist_ok=True)
    candidates_file.write_text("", encoding="utf-8")
    manifest = _new_manifest(config, selected)
    _write_json(manifest_file, manifest)
    return {}, manifest


def _load_sft_adapter(config: DpoMiningConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("DPO candidate generation requires CUDA")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise RuntimeError("bitsandbytes is required for 4-bit candidate generation") from error

    adapter_config = PeftConfig.from_pretrained(str(config.model.adapter_path))
    base_model_name = adapter_config.base_model_name_or_path
    if not base_model_name:
        raise ValueError("The SFT adapter config does not identify its base model")
    tokenizer_source = (
        config.model.adapter_path
        if (config.model.adapter_path / "tokenizer_config.json").is_file()
        else Path(base_model_name)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        use_fast=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    device = torch.cuda.current_device()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(config.model.adapter_path),
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, str(base_model_name)


def _completion_length_and_cap(
    token_ids: list[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_new_tokens: int,
) -> tuple[int, bool]:
    for index, token_id in enumerate(token_ids):
        if eos_token_id is not None and token_id == eos_token_id:
            return index + 1, False
        if pad_token_id is not None and token_id == pad_token_id:
            return index, False
    length = len(token_ids)
    return length, length >= max_new_tokens


def _generate_candidates(
    config: DpoMiningConfig,
    selected: list[MiningExample],
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    existing, manifest = _prepare_generation_artifacts(
        config,
        selected,
        resume=resume,
        overwrite=overwrite,
    )
    unexpected_ids = set(existing) - {example.question_id for example in selected}
    if unexpected_ids:
        raise ValueError(f"Candidate file contains unexpected question IDs: {unexpected_ids}")
    for question_id, group in existing.items():
        candidates = group.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != config.sampling.num_candidates:
            raise ValueError(
                f"question_id={question_id} does not contain the configured candidate count"
            )

    if len(existing) == len(selected):
        print("All selected question groups are already generated", flush=True)
        manifest.update(
            {
                "status": "generation_complete",
                "generation_completed_at_utc": datetime.now(UTC).isoformat(),
                "generated_questions": len(existing),
                "generated_candidates": (
                    len(existing) * config.sampling.num_candidates
                ),
            }
        )
        _write_json(config.output.manifest_file, manifest)
        return manifest

    model, tokenizer, base_model_name = _load_sft_adapter(config)
    started = time.monotonic()
    generated_questions = len(existing)
    num_candidates = config.sampling.num_candidates
    batches = list(_batch(selected, config.sampling.prompt_batch_size))
    for batch_index, batch_examples in enumerate(batches):
        missing_examples = [
            example for example in batch_examples if example.question_id not in existing
        ]
        if not missing_examples:
            continue
        batch_seed = config.sampling.seed + batch_index
        set_seed(batch_seed)
        rendered_prompts = [
            tokenizer.apply_chat_template(
                list(example.prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in missing_examples
        ]
        inputs = tokenizer(
            rendered_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_lengths = [int(value) for value in inputs["attention_mask"].sum(dim=1)]
        overlength = [
            (example.question_id, length)
            for example, length in zip(missing_examples, input_lengths, strict=True)
            if length > config.data.max_input_length
        ]
        if overlength:
            raise ValueError(
                f"Mining prompts exceed max_input_length={config.data.max_input_length}: "
                f"{overlength}"
            )

        inputs = inputs.to(model.device)
        input_width = int(inputs["input_ids"].shape[1])
        batch_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=config.sampling.temperature,
                top_p=config.sampling.top_p,
                num_return_sequences=num_candidates,
                max_new_tokens=config.sampling.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        batch_elapsed = time.monotonic() - batch_started
        generated_ids = generated[:, input_width:]
        generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for example_index, example in enumerate(missing_examples):
            candidates: list[dict[str, Any]] = []
            start = example_index * num_candidates
            for candidate_index in range(num_candidates):
                flat_index = start + candidate_index
                token_ids = [int(value) for value in generated_ids[flat_index].tolist()]
                generated_tokens, hit_cap = _completion_length_and_cap(
                    token_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=config.sampling.max_new_tokens,
                )
                candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "generation": generations[flat_index],
                        "input_tokens": input_lengths[example_index],
                        "generated_tokens": generated_tokens,
                        "hit_max_new_tokens": hit_cap,
                        "batch_seed": batch_seed,
                        "generation_seconds": round(
                            batch_elapsed / len(missing_examples), 6
                        ),
                    }
                )
            group = {
                "source_index": example.source_index,
                "question_id": example.question_id,
                "db_id": example.db_id,
                "prompt": list(example.prompt),
                "gold_sql": example.gold_sql,
                "candidates": candidates,
            }
            _append_jsonl(config.output.candidates_file, group)
            existing[example.question_id] = group
            generated_questions += 1
        print(
            f"Generated batch {batch_index + 1}/{len(batches)}: "
            f"{generated_questions}/{len(selected)} questions",
            flush=True,
        )

    elapsed = time.monotonic() - started
    manifest.update(
        {
            "status": "generation_complete",
            "generation_completed_at_utc": datetime.now(UTC).isoformat(),
            "resolved_base_model": base_model_name,
            "generated_questions": len(existing),
            "generated_candidates": len(existing) * num_candidates,
            "generation_elapsed_seconds_this_run": round(elapsed, 6),
        }
    )
    _write_json(config.output.manifest_file, manifest)
    return manifest


def _prepare_build_artifacts(
    config: DpoMiningConfig,
    *,
    resume: bool,
    overwrite: bool,
) -> dict[int, dict[str, Any]]:
    verified_file = config.output.verified_candidates_file
    derived_files = (
        config.output.train_file,
        config.output.validation_file,
        config.output.report_file,
    )
    if overwrite:
        verified_file.parent.mkdir(parents=True, exist_ok=True)
        verified_file.write_text("", encoding="utf-8")
        return {}
    if resume:
        if not verified_file.exists():
            verified_file.parent.mkdir(parents=True, exist_ok=True)
            verified_file.write_text("", encoding="utf-8")
            return {}
        rows = _read_jsonl(verified_file)
        existing = {int(row["question_id"]): row for row in rows}
        if len(existing) != len(rows):
            raise ValueError("Duplicate question groups in verified_candidates.jsonl")
        return existing
    existing_outputs = [path for path in (verified_file, *derived_files) if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Build artifacts already exist: {paths}; use --resume")
    verified_file.parent.mkdir(parents=True, exist_ok=True)
    verified_file.write_text("", encoding="utf-8")
    return {}


def _build_pairs(
    config: DpoMiningConfig,
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if not config.output.candidates_file.is_file():
        raise FileNotFoundError(
            f"Candidate file does not exist: {config.output.candidates_file}"
        )
    raw_groups = _read_jsonl(config.output.candidates_file)
    if not raw_groups:
        raise ValueError("Candidate file is empty")
    existing = _prepare_build_artifacts(
        config,
        resume=resume,
        overwrite=overwrite,
    )
    expected_ids = {int(group["question_id"]) for group in raw_groups}
    unexpected_ids = set(existing) - expected_ids
    if unexpected_ids:
        raise ValueError(f"Verified file contains unexpected question IDs: {unexpected_ids}")

    database_paths: dict[str, Path] = {}
    started = time.monotonic()
    for index, raw_group in enumerate(raw_groups, start=1):
        question_id = int(raw_group["question_id"])
        if question_id in existing:
            continue
        db_id = str(raw_group["db_id"])
        if db_id not in database_paths:
            database_paths[db_id] = resolve_database(config.execution.db_root, db_id)
        verified = verify_candidate_group(
            raw_group,
            database_paths[db_id],
            timeout_seconds=config.execution.timeout_seconds,
            max_result_rows=config.execution.max_result_rows,
        )
        _append_jsonl(config.output.verified_candidates_file, verified)
        existing[question_id] = verified
        if index % 25 == 0 or index == len(raw_groups):
            print(f"Verified {len(existing)}/{len(raw_groups)} question groups", flush=True)

    verified_groups = [existing[int(group["question_id"])] for group in raw_groups]
    pairs: list[dict[str, Any]] = []
    pair_outcomes: Counter[str] = Counter()
    for group in verified_groups:
        pair, outcome = select_preference_pair(
            group,
            include_execution_error_rejected=(
                config.pairing.include_execution_error_rejected
            ),
        )
        pair_outcomes[outcome] += 1
        if pair is not None:
            pairs.append(pair)
    train_pairs, validation_pairs = split_preference_pairs(
        pairs,
        validation_ratio=config.pairing.validation_ratio,
        seed=config.pairing.seed,
    )
    _write_jsonl(config.output.train_file, train_pairs)
    _write_jsonl(config.output.validation_file, validation_pairs)
    report = summarize_verified_groups(
        verified_groups,
        pair_outcomes,
        pairs,
        train_pairs,
        validation_pairs,
    )
    report.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "verification_elapsed_seconds_this_run": round(
                time.monotonic() - started, 6
            ),
            "candidates_file": str(config.output.candidates_file),
            "verified_candidates_file": str(config.output.verified_candidates_file),
            "train_file": str(config.output.train_file),
            "validation_file": str(config.output.validation_file),
            "gold_execution_policy": "inline_once_per_question_no_prevalidation_pass",
        }
    )
    _write_json(config.output.report_file, report)
    manifest = _read_json(config.output.manifest_file)
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "mining_report": report,
        }
    )
    _write_json(config.output.manifest_file, manifest)
    return report


def main() -> None:
    args = _parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = DpoMiningConfig.from_yaml(args.config).with_overrides(
        adapter_path=args.adapter_path,
        db_root=args.db_root,
        output_dir=args.output_dir,
        sample_size=args.limit,
    )
    config.validate_inputs(
        require_adapter=(not args.validate_only and args.stage in {"generate", "all"}),
        require_databases=(not args.validate_only and args.stage in {"build", "all"}),
    )
    loaded_examples = load_mining_examples(config.data.path)
    loaded_question_ids = {example.question_id for example in loaded_examples}
    missing_exclusions = set(config.data.exclude_question_ids) - loaded_question_ids
    if missing_exclusions:
        raise ValueError(
            f"Configured excluded question IDs are absent from the source: {missing_exclusions}"
        )
    excluded_question_ids = set(config.data.exclude_question_ids)
    source_examples = [
        example
        for example in loaded_examples
        if example.question_id not in excluded_question_ids
    ]
    selected = select_mining_examples(
        source_examples,
        sample_size=config.data.sample_size,
        seed=config.data.seed,
    )
    selection_summary = {
        "loaded_questions": len(loaded_examples),
        "excluded_question_ids": list(config.data.exclude_question_ids),
        "source_questions": len(source_examples),
        "selected_questions": len(selected),
        "source_databases": len({example.db_id for example in source_examples}),
        "selected_databases": len({example.db_id for example in selected}),
        "selected_question_ids_sha256": selected_question_ids_sha256(selected),
        "output_dir": str(config.output.directory),
    }
    if args.validate_only:
        print(json.dumps(selection_summary, ensure_ascii=False, indent=2))
        return

    if args.stage in {"generate", "all"}:
        _generate_candidates(
            config,
            selected,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.stage in {"build", "all"}:
        if not config.output.manifest_file.is_file():
            raise FileNotFoundError(
                f"Mining manifest does not exist: {config.output.manifest_file}"
            )
        manifest = _read_json(config.output.manifest_file)
        if manifest.get("fingerprint") != _selection_fingerprint(config, selected):
            raise ValueError(
                "Candidate manifest fingerprint does not match the current mining config"
            )
        report = _build_pairs(
            config,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
