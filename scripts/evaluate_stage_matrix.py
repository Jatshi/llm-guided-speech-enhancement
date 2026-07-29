from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from lse_v2.config import load_config, resolve_path, set_global_seed
from lse_v2.contracts import GRPO_SCHEMA, validate_alignment_record
from lse_v2.evaluation import evaluate_rewards
from lse_v2.io import read_jsonl, utc_now, write_json_atomic, write_jsonl


def stage_candidates(config: dict[str, Any]) -> list[tuple[str, Path | None]]:
    stages = config["training"]["stages"]
    return [
        ("base", None),
        ("sft", resolve_path(config, stages["sft"]["output_dir"]) / "final"),
        ("dpo", resolve_path(config, stages["dpo"]["output_dir"]) / "final"),
        ("grpo", resolve_path(config, stages["grpo"]["output_dir"]) / "final"),
    ]


def _generate_stage(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    adapter_path: Path | None,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, float | int | str | None]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = str(config["model"].get("dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        torch_dtype=dtype,
        device_map=config["model"].get("device_map", "auto"),
    )
    model = PeftModel.from_pretrained(base, str(adapter_path)) if adapter_path else base
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions: list[dict[str, str]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        input_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
        )
        predictions.extend(
            {
                "sample_id": row["sample_id"],
                "response": response.strip(),
            }
            for row, response in zip(batch, decoded, strict=True)
        )
    elapsed = time.perf_counter() - started
    peak_mib = (
        float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
    )
    del generated, inputs, model, base, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, {
        "samples": len(predictions),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(predictions) / elapsed if elapsed else 0.0,
        "mean_latency_seconds_per_sample": elapsed / len(predictions) if predictions else None,
        "peak_memory_allocated_mib": peak_mib,
        "adapter": str(adapter_path) if adapter_path else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate base, SFT, DPO and GRPO on the same held-out records"
    )
    parser.add_argument("--config", default="configs/autodl_4090.json")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", default="outputs/v2/stage_matrix")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    if args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("max-samples and batch-size must be positive")
    config = load_config(args.config)
    set_global_seed(int(config["project"]["seed"]))
    dataset_path = (
        Path(args.dataset).expanduser().resolve()
        if args.dataset
        else resolve_path(config, config["data"]["grpo_eval"])
    )
    all_rows = read_jsonl(dataset_path)
    if not all_rows:
        raise ValueError(f"stage-matrix dataset is empty: {dataset_path}")
    sample_count = min(args.max_samples, len(all_rows))
    rng = random.Random(int(config["project"]["seed"]))
    selected_indices = sorted(rng.sample(range(len(all_rows)), sample_count))
    rows = [all_rows[index] for index in selected_indices]
    for row in rows:
        validate_alignment_record(row)
        if row["schema_version"] != GRPO_SCHEMA:
            raise ValueError("stage matrix requires lse.grpo.v2 records")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else config.get("evaluation", {}).get("max_new_tokens", 256)
    )
    matrix: dict[str, Any] = {
        "schema_version": "lse.stage_matrix.v2",
        "created_at": utc_now(),
        "config": str(Path(args.config).resolve()),
        "dataset": str(dataset_path),
        "dataset_records": len(all_rows),
        "records": len(rows),
        "selection": {
            "method": "uniform_without_replacement",
            "seed": int(config["project"]["seed"]),
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")
            ).hexdigest(),
        },
        "stages": {},
    }
    write_json_atomic(output_dir / "matrix_status.json", matrix)
    for stage, adapter_path in stage_candidates(config):
        if adapter_path is not None and not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{stage} adapter not found: {adapter_path}")
        predictions, runtime = _generate_stage(
            config=config,
            rows=rows,
            adapter_path=adapter_path,
            batch_size=args.batch_size,
            max_new_tokens=max_new_tokens,
        )
        prediction_path = output_dir / f"{stage}_predictions.jsonl"
        report_path = output_dir / f"{stage}_reward_report.json"
        write_jsonl(prediction_path, predictions)
        reward_report = evaluate_rewards(
            dataset_path,
            predictions_path=prediction_path,
            include_ablations=True,
        )
        write_json_atomic(report_path, reward_report)
        matrix["stages"][stage] = {
            "runtime": runtime,
            "predictions": str(prediction_path),
            "reward_report": str(report_path),
            "metrics": reward_report["metrics"],
        }
        write_json_atomic(output_dir / "matrix_status.json", matrix)
    matrix["finished_at"] = utc_now()
    matrix["status"] = "complete"
    write_json_atomic(output_dir / "stage_matrix.json", matrix)
    write_json_atomic(output_dir / "matrix_status.json", matrix)
    print(json.dumps(matrix, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
