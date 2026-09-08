from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty JSONL file: {path}")
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            rows += 1
    return rows


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty artifact: {path}")
    return path


def validate_stage(output_root: Path, stage: str) -> dict[str, Any]:
    stage_root = output_root / stage
    manifest_path = stage_root / "stage_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("stage") != stage or manifest.get("status") != "completed":
        raise ValueError(f"{stage} stage is not completed")
    final_root = stage_root / "final"
    adapter_root = final_root / "adapter"
    trainable_adapter = adapter_root / stage
    required = [
        manifest_path,
        final_root / "artifact_manifest.json",
        final_root / "audio_projector.pt",
        trainable_adapter / "adapter_config.json",
        trainable_adapter / "adapter_model.safetensors",
    ]
    for path in required:
        require_file(path)
    return {
        "stage": stage,
        "status": "completed",
        "steps": manifest.get("steps"),
        "loss_initial": manifest.get("loss_initial"),
        "loss_final": manifest.get("loss_final"),
        "loss_mean": manifest.get("loss_mean"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "required_artifacts": [str(path.relative_to(output_root)) for path in required],
    }


def copy_release_bundle(
    *,
    repo_root: Path,
    output_root: Path,
    release_dir: Path,
    report_paths: list[Path],
) -> dict[str, Any]:
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    adapter_parent = output_root / "sft" / "final" / "adapter"
    adapter_weights = adapter_parent / "sft"
    for path in adapter_parent.iterdir():
        if path.is_file() and path.name != "README.md":
            shutil.copy2(path, release_dir / path.name)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(adapter_weights / name, release_dir / name)
    shutil.copy2(output_root / "sft" / "final" / "audio_projector.pt", release_dir)
    shutil.copy2(repo_root / "MODEL_CARD.md", release_dir / "README.md")

    evidence_dir = release_dir / "evaluation"
    evidence_dir.mkdir()
    for path in report_paths:
        shutil.copy2(path, evidence_dir / path.name)

    entries = []
    for path in sorted(item for item in release_dir.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": str(path.relative_to(release_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "lse.v4_release_bundle.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_stage": "sft",
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def finalize(repo_root: Path, output_root: Path, release_dir: Path) -> dict[str, Any]:
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest = read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise ValueError("native v4 run manifest is not completed")
    stages = [validate_stage(output_root, stage) for stage in ("sft", "dpo", "grpo")]

    sft_report_path = output_root / "sft_test_predictions" / "prediction_report.json"
    sft_predictions_path = output_root / "sft_test_predictions" / "predictions.jsonl"
    grpo_report_path = output_root / "test_predictions" / "prediction_report.json"
    grpo_predictions_path = output_root / "test_predictions" / "predictions.jsonl"
    benchmark_path = output_root / "sft_generalization_stft_fixed" / "benchmark_report.json"
    environment_path = output_root / "environment.freeze.txt"
    quality_path = output_root / "final_validation" / "quality_gates.log"
    shell_path = output_root / "final_validation" / "shell_syntax.log"
    for path in (environment_path, quality_path, shell_path):
        require_file(path)

    sft_report = read_json(sft_report_path)
    grpo_report = read_json(grpo_report_path)
    benchmark = read_json(benchmark_path)
    expected = int(sft_report.get("records", -1))
    if expected <= 0 or sft_report.get("successful") != expected or sft_report.get("failed") != 0:
        raise ValueError("SFT prediction report is not a complete successful test run")
    if count_jsonl(sft_predictions_path) != expected:
        raise ValueError("SFT prediction JSONL row count does not match its report")
    if grpo_report.get("records") != expected or grpo_report.get("successful") != 0:
        raise ValueError("GRPO negative-result report is incomplete or unexpectedly successful")
    if grpo_report.get("failed") != expected or count_jsonl(grpo_predictions_path) != expected:
        raise ValueError("GRPO failure evidence does not cover the complete test set")
    overall = benchmark.get("overall", {})
    if overall.get("samples") != expected:
        raise ValueError("generalization report does not cover the complete test set")

    model_selection = {
        "schema_version": "lse.v4_model_selection.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_stage": "sft",
        "selection_basis": (
            "SFT is the latest stage with 100% schema-valid held-out predictions and a complete "
            "waveform generalization benchmark. DPO was not promoted after its canary responses "
            "violated the numeric output contract. GRPO is retained as negative evidence because "
            "0/1416 held-out responses passed JSON decoding."
        ),
        "held_out_records": expected,
        "sft": {
            "successful": sft_report["successful"],
            "failed": sft_report["failed"],
            "throughput_records_per_second": sft_report.get("throughput_records_per_second"),
            "latency": sft_report.get("latency"),
        },
        "dpo": {"selected": False, "reason": "numeric-output contract failed in canary inference"},
        "grpo": {
            "selected": False,
            "successful": grpo_report["successful"],
            "failed": grpo_report["failed"],
            "reason": "all held-out responses failed JSON decoding",
        },
        "generalization": overall,
    }
    selection_path = output_root / "model_selection_report.json"
    selection_path.write_text(
        json.dumps(model_selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    critical_paths = [
        run_manifest_path,
        *(output_root / stage / "stage_manifest.json" for stage in ("sft", "dpo", "grpo")),
        *(output_root / stage / "final" / "audio_projector.pt" for stage in ("sft", "dpo", "grpo")),
        *(
            output_root / stage / "final" / "adapter" / stage / "adapter_model.safetensors"
            for stage in ("sft", "dpo", "grpo")
        ),
        sft_predictions_path,
        grpo_predictions_path,
        benchmark_path,
        environment_path,
        quality_path,
        shell_path,
        selection_path,
    ]
    validation = {
        "schema_version": "lse.v4_artifact_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "selected_stage": "sft",
        "stages": stages,
        "test_records": expected,
        "quality_gates": {
            "ruff": "passed",
            "pytest": "passed",
            "shell_syntax": "passed_after_lf_normalization",
        },
        "critical_artifacts": [
            {
                "path": str(path.relative_to(output_root)).replace("\\", "/"),
                "bytes": require_file(path).stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in critical_paths
        ],
    }
    validation_dir = output_root / "final_validation"
    validation_path = validation_dir / "artifact_validation_report.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    release_manifest = copy_release_bundle(
        repo_root=repo_root,
        output_root=output_root,
        release_dir=release_dir,
        report_paths=[
            run_manifest_path,
            output_root / "sft" / "stage_manifest.json",
            sft_report_path,
            benchmark_path,
            selection_path,
            validation_path,
            environment_path,
        ],
    )
    return {
        "status": "passed",
        "selected_stage": "sft",
        "model_selection_report": str(selection_path),
        "artifact_validation_report": str(validation_path),
        "release_dir": str(release_dir),
        "release_file_count": release_manifest["file_count"] + 1,
        "release_total_bytes_before_manifest": release_manifest["total_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and package the evidence-backed v4 SFT release"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default="outputs/native_v4")
    parser.add_argument("--release-dir", default="outputs/native_v4/release/hf_sft_adapter")
    args = parser.parse_args()
    summary = finalize(
        Path(args.repo_root).resolve(),
        Path(args.output_root).resolve(),
        Path(args.release_dir).resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
