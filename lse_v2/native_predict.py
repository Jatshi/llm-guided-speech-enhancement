"""Generate native-audio prescriptions for held-out manifests with latency evidence."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, utc_now, write_json_atomic, write_jsonl
from .native_pipeline import load_native_records


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64) * 1000
    return {
        "samples": len(values),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
    }


def _chunks(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("batch_size must be positive")
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def _embedding_paths(index_path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in read_jsonl(index_path):
        sample_id = str(row.get("sample_id", ""))
        embedding_path = str(row.get("embedding_path", ""))
        if not sample_id or not embedding_path:
            raise ValueError("embedding index rows require sample_id and embedding_path")
        if sample_id in result:
            raise ValueError(f"duplicate sample_id in embedding index: {sample_id}")
        result[sample_id] = embedding_path
    return result


def _prediction(row: dict[str, Any], raw_text: str, elapsed: float, split: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("native policy did not return a JSON object")
        prescription = json.dumps(parsed, ensure_ascii=False)
        status, error = "ok", None
    except (ValueError, json.JSONDecodeError) as exc:
        prescription, status, error = None, "failed", f"{type(exc).__name__}: {exc}"
    return {
        "sample_id": row["sample_id"],
        "split": split,
        "status": status,
        "prescription": prescription,
        "raw_response": raw_text,
        "error": error,
        "latency_seconds": elapsed,
    }


def predict_manifest(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    stage_dir: str | Path,
    whisper_model: str,
    language_model: str,
    prefix_tokens: int,
    split: str = "test",
    fail_fast: bool = False,
    embedding_index: str | Path | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 192,
    temperature: float = 0.2,
    resume: bool = False,
) -> dict[str, Any]:
    from .native_inference import NativeAudioPlanner

    rows = [row for row in load_native_records(Path(manifest)) if row["split"] == split]
    if not rows:
        raise ValueError(f"manifest has no {split} records")
    if batch_size > 1 and embedding_index is None:
        raise ValueError("batch_size > 1 requires --embedding-index")
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    partial_path = target / "predictions.partial.jsonl"
    predictions = read_jsonl(partial_path) if resume and partial_path.exists() else []
    completed_ids = {str(row["sample_id"]) for row in predictions}
    pending = [row for row in rows if str(row["sample_id"]) not in completed_ids]
    cache = _embedding_paths(embedding_index) if embedding_index is not None else None
    if cache is not None:
        missing = [str(row["sample_id"]) for row in pending if str(row["sample_id"]) not in cache]
        if missing:
            raise ValueError(f"embedding index is missing {len(missing)} requested samples")
    planner = NativeAudioPlanner(
        stage_dir,
        whisper_model=whisper_model,
        language_model=language_model,
        prefix_tokens=prefix_tokens,
        cache_only=cache is not None,
    )
    started_run = time.perf_counter()
    mode = "a" if predictions else "w"
    with partial_path.open(mode, encoding="utf-8", newline="\n") as partial:
        for batch in _chunks(pending, batch_size):
            started = time.perf_counter()
            if cache is not None:
                raw_texts = planner.generate_cached_batch(
                    [cache[str(row["sample_id"])] for row in batch],
                    [str(row["prompt_text"]) for row in batch],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                elapsed = (time.perf_counter() - started) / len(batch)
                batch_predictions = [
                    _prediction(row, raw, elapsed, split)
                    for row, raw in zip(batch, raw_texts, strict=True)
                ]
            else:
                batch_predictions = []
                row = batch[0]
                try:
                    prescription = planner.plan_audio(
                        row["audio_path"],
                        row["prompt_text"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    status, error = "ok", None
                except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                    if fail_fast:
                        raise
                    prescription, status, error = None, "failed", f"{type(exc).__name__}: {exc}"
                batch_predictions.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": split,
                        "status": status,
                        "prescription": prescription,
                        "raw_response": prescription,
                        "error": error,
                        "latency_seconds": time.perf_counter() - started,
                    }
                )
            failed = [row for row in batch_predictions if row["status"] == "failed"]
            if fail_fast and failed:
                raise ValueError(str(failed[0]["error"]))
            for prediction in batch_predictions:
                partial.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
            partial.flush()
            predictions.extend(batch_predictions)
    elapsed_run = time.perf_counter() - started_run
    by_id = {str(row["sample_id"]): row for row in predictions}
    predictions = [by_id[str(row["sample_id"])] for row in rows]
    latencies = [float(row["latency_seconds"]) for row in predictions]
    write_jsonl(target / "predictions.jsonl", predictions)
    report = {
        "schema_version": "lse.native_predictions.v1",
        "created_at": utc_now(),
        "manifest": str(Path(manifest).resolve()),
        "stage_dir": str(Path(stage_dir).resolve()),
        "split": split,
        "records": len(rows),
        "successful": sum(row["status"] == "ok" for row in predictions),
        "failed": sum(row["status"] == "failed" for row in predictions),
        "latency": latency_summary(latencies),
        "wall_seconds_this_invocation": elapsed_run,
        "throughput_records_per_second": len(pending) / elapsed_run if elapsed_run else None,
        "batch_size": batch_size,
        "embedding_index": str(Path(embedding_index).resolve()) if embedding_index else None,
        "predictions": str(target / "predictions.jsonl"),
    }
    write_json_atomic(target / "prediction_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    parser.add_argument("--language-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="test")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--embedding-index", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    report = predict_manifest(
        args.manifest,
        args.output_dir,
        stage_dir=args.stage_dir,
        whisper_model=args.whisper_model,
        language_model=args.language_model,
        prefix_tokens=args.prefix_tokens,
        split=args.split,
        fail_fast=args.fail_fast,
        embedding_index=args.embedding_index,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
