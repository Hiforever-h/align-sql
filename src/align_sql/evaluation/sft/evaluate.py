"""CLI for deterministic base/SFT generation and SQL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from align_sql.evaluation.sft.config import SftEvalConfig
from align_sql.evaluation.sft.data import EvalExample, load_eval_examples
from align_sql.evaluation.sft.execution import compare_execution, resolve_database
from align_sql.evaluation.sft.metrics import analyze_generation, summarize_predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AlignSQL base or SFT model")
    parser.add_argument("--config", default="configs/eval_sft.yaml")
    parser.add_argument("--model-mode", choices=("base", "adapter"))
    parser.add_argument("--base-model")
    parser.add_argument("--adapter")
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--db-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--gold-as-prediction",
        action="store_true",
        help="Bypass model loading and evaluate gold SQL as predictions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing evaluation artifacts without deleting the output directory",
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


def _batch(examples: list[EvalExample], batch_size: int) -> Iterator[list[EvalExample]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def _load_model(config: SftEvalConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("Model generation requires CUDA")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise RuntimeError("bitsandbytes is required for 4-bit evaluation") from error

    adapter_path = config.model.adapter_path
    base_model_name = config.model.base_model_name_or_path
    if config.model.mode == "adapter":
        if adapter_path is None:
            raise ValueError("adapter mode requires adapter_path")
        adapter_config = PeftConfig.from_pretrained(str(adapter_path))
        resolved_base_model = adapter_config.base_model_name_or_path
        if not resolved_base_model:
            raise ValueError("The adapter config does not identify its base model")
        base_model_name = resolved_base_model

    tokenizer_source = (
        adapter_path
        if (
            config.model.mode == "adapter"
            and adapter_path is not None
            and (adapter_path / "tokenizer_config.json").is_file()
        )
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
    model = base_model
    if config.model.mode == "adapter":
        if adapter_path is None:
            raise ValueError("adapter mode requires adapter_path")
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False,
        )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, base_model_name


def _generate_predictions(
    examples: list[EvalExample],
    config: SftEvalConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, tokenizer, base_model_name = _load_model(config)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    set_seed(config.generation.seed)

    for batch_index, batch_examples in enumerate(
        _batch(examples, config.generation.batch_size),
        start=1,
    ):
        rendered_prompts = [
            tokenizer.apply_chat_template(
                list(example.prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in batch_examples
        ]
        inputs = tokenizer(
            rendered_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        overlength = [
            (example.question_id, int(length))
            for example, length in zip(batch_examples, input_lengths, strict=True)
            if int(length) > config.data.max_input_length
        ]
        if overlength:
            raise ValueError(
                f"Evaluation prompts exceed max_input_length={config.data.max_input_length}: "
                f"{overlength}"
            )

        inputs = inputs.to(model.device)
        input_width = inputs["input_ids"].shape[1]
        batch_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=config.generation.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        batch_elapsed = time.monotonic() - batch_started
        generated_ids = generated[:, input_width:]
        generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for example, generation, input_length, output_ids in zip(
            batch_examples,
            generations,
            input_lengths,
            generated_ids,
            strict=True,
        ):
            generated_tokens = int(
                (output_ids != tokenizer.pad_token_id).sum().detach().cpu().item()
            )
            records.append(
                _prediction_record(
                    example,
                    generation=generation,
                    input_tokens=int(input_length),
                    generated_tokens=generated_tokens,
                    generation_seconds=batch_elapsed / len(batch_examples),
                )
            )
        print(
            f"Generated batch {batch_index}: {len(records)}/{len(examples)} examples",
            flush=True,
        )

    elapsed = time.monotonic() - started
    return records, {
        "model_mode": config.model.mode,
        "base_model_name_or_path": base_model_name,
        "elapsed_seconds": round(elapsed, 6),
        "examples_per_second": round(len(examples) / elapsed, 6),
    }


def _gold_predictions(examples: list[EvalExample]) -> list[dict[str, Any]]:
    return [
        _prediction_record(
            example,
            generation=f"```sql\n{example.gold_sql.rstrip(';')}\n```",
            input_tokens=0,
            generated_tokens=0,
            generation_seconds=0.0,
        )
        for example in examples
    ]


def _prediction_record(
    example: EvalExample,
    *,
    generation: str,
    input_tokens: int,
    generated_tokens: int,
    generation_seconds: float,
) -> dict[str, Any]:
    analysis = analyze_generation(generation, example.gold_sql)
    return {
        "index": example.index,
        "question_id": example.question_id,
        "db_id": example.db_id,
        "prompt": list(example.prompt),
        "gold_sql": example.gold_sql,
        "generation": generation,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generation_seconds": round(generation_seconds, 6),
        **analysis.to_dict(),
        "execution": None,
    }


def _add_execution(
    records: list[dict[str, Any]],
    config: SftEvalConfig,
) -> None:
    db_root = config.execution.db_root
    if db_root is None:
        return
    database_paths = {
        db_id: resolve_database(db_root, db_id)
        for db_id in sorted({str(record["db_id"]) for record in records})
    }
    for index, record in enumerate(records, start=1):
        candidate_sql = record["extracted_sql"]
        if candidate_sql is None:
            record["execution"] = {
                "match": False,
                "order_sensitive": None,
                "candidate": {
                    "status": "missing_sql",
                    "row_count": 0,
                    "digest": None,
                    "elapsed_seconds": 0.0,
                    "error": "No SQL could be extracted from the generation",
                },
                "gold": None,
            }
        else:
            record["execution"] = compare_execution(
                database_paths[str(record["db_id"])],
                candidate_sql,
                str(record["gold_sql"]),
                timeout_seconds=config.execution.timeout_seconds,
                max_result_rows=config.execution.max_result_rows,
            )
        if index % 25 == 0 or index == len(records):
            print(f"Executed {index}/{len(records)} predictions", flush=True)


def _ensure_output(output_dir: Path, overwrite: bool) -> None:
    artifacts = (
        output_dir / "predictions.jsonl",
        output_dir / "metrics.json",
        output_dir / "eval_manifest.json",
    )
    existing = [path for path in artifacts if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Evaluation artifacts already exist: {paths}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def _manifest(
    config: SftEvalConfig,
    *,
    examples: list[EvalExample],
    gold_as_prediction: bool,
    generation_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config.source_path),
        "gold_as_prediction": gold_as_prediction,
        "model": {
            **asdict(config.model),
            "adapter_path": (
                str(config.model.adapter_path)
                if config.model.adapter_path is not None
                else None
            ),
        },
        "data": {
            **asdict(config.data),
            "path": str(config.data.path),
            "sha256": _sha256_file(config.data.path),
            "loaded_examples": len(examples),
        },
        "generation": {**asdict(config.generation), **generation_info},
        "execution": {
            **asdict(config.execution),
            "db_root": (
                str(config.execution.db_root)
                if config.execution.db_root is not None
                else None
            ),
            "metric_scope": "internal_read_only_sqlite_not_official_bird_leaderboard",
        },
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
    }


def main() -> None:
    args = _parse_args()
    config = SftEvalConfig.from_yaml(args.config).with_overrides(
        model_mode=args.model_mode,
        base_model_name_or_path=args.base_model,
        adapter_path=args.adapter,
        data_path=args.data,
        output_dir=args.output_dir,
        db_root=args.db_root,
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    config.validate_inputs(require_model=not args.gold_as_prediction)
    _ensure_output(config.output.directory, args.overwrite)
    examples = load_eval_examples(config.data.path, limit=config.data.limit)

    if args.gold_as_prediction:
        records = _gold_predictions(examples)
        generation_info: dict[str, Any] = {"mode": "gold_self_check"}
    else:
        records, generation_info = _generate_predictions(examples, config)
        generation_info["mode"] = "greedy"
    _add_execution(records, config)
    metrics = summarize_predictions(records)
    manifest = _manifest(
        config,
        examples=examples,
        gold_as_prediction=args.gold_as_prediction,
        generation_info=generation_info,
    )
    _write_jsonl(config.output.directory / "predictions.jsonl", records)
    _write_json(config.output.directory / "metrics.json", metrics)
    _write_json(config.output.directory / "eval_manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
