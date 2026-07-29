from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PENDING_MODEL_CARD_MARKER = "FINAL_EVAL_TABLE"
EXPECTED_RELEASE_EVAL_RECORDS = 200


def validate_release(
    artifact_dir: Path,
    stage_manifest_path: Path,
    model_card_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    if not artifact_dir.is_dir():
        raise ValueError(f"adapter directory not found: {artifact_dir}")
    required_files = ("adapter_config.json", "adapter_model.safetensors")
    missing_files = [name for name in required_files if not (artifact_dir / name).is_file()]
    if missing_files:
        raise ValueError(f"missing adapter artifacts: {', '.join(missing_files)}")
    if not stage_manifest_path.is_file():
        raise ValueError(f"stage manifest not found: {stage_manifest_path}")
    manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "grpo" or manifest.get("status") != "complete":
        raise ValueError("stage manifest must prove a completed GRPO run")
    if not model_card_path.is_file():
        raise ValueError(f"model card not found: {model_card_path}")
    model_card = model_card_path.read_text(encoding="utf-8")
    if PENDING_MODEL_CARD_MARKER in model_card:
        raise ValueError("model card still contains the pending final-evaluation marker")
    if "license: mit" not in model_card:
        raise ValueError("model card must declare the repository's MIT adapter license")
    if not evaluation_path.is_file():
        raise ValueError(f"stage-matrix evaluation not found: {evaluation_path}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "complete":
        raise ValueError("stage-matrix evaluation must be complete")
    if set(evaluation.get("stages", {})) != {"base", "sft", "dpo", "grpo"}:
        raise ValueError("stage-matrix evaluation must contain base, SFT, DPO, and GRPO")
    if evaluation.get("records") != EXPECTED_RELEASE_EVAL_RECORDS:
        raise ValueError(
            "stage-matrix evaluation must contain exactly "
            f"{EXPECTED_RELEASE_EVAL_RECORDS} held-out records"
        )
    selection = evaluation.get("selection", {})
    selection_sha256 = selection.get("sample_ids_sha256")
    if (
        selection.get("method") != "uniform_without_replacement"
        or selection.get("seed") != 42
        or not isinstance(selection_sha256, str)
        or len(selection_sha256) != 64
    ):
        raise ValueError("stage-matrix selection provenance is incomplete")
    required_evaluation_files = [
        evaluation_path.parent / f"{stage}_{suffix}"
        for stage in ("base", "sft", "dpo", "grpo")
        for suffix in ("predictions.jsonl", "reward_report.json")
    ]
    missing_evaluation_files = [
        path.name for path in required_evaluation_files if not path.is_file()
    ]
    if missing_evaluation_files:
        raise ValueError("missing stage-matrix artifacts: " + ", ".join(missing_evaluation_files))
    for stage in ("base", "sft", "dpo", "grpo"):
        predictions_path = evaluation_path.parent / f"{stage}_predictions.jsonl"
        prediction_rows = sum(
            bool(line.strip()) for line in predictions_path.read_text(encoding="utf-8").splitlines()
        )
        if prediction_rows != EXPECTED_RELEASE_EVAL_RECORDS:
            raise ValueError(
                f"{stage} predictions must contain exactly "
                f"{EXPECTED_RELEASE_EVAL_RECORDS} non-empty rows"
            )
        reward_report_path = evaluation_path.parent / f"{stage}_reward_report.json"
        reward_report = json.loads(reward_report_path.read_text(encoding="utf-8"))
        if reward_report.get("num_samples") != EXPECTED_RELEASE_EVAL_RECORDS:
            raise ValueError(
                f"{stage} reward report must cover exactly {EXPECTED_RELEASE_EVAL_RECORDS} samples"
            )
    return {
        "artifact_dir": str(artifact_dir),
        "stage_manifest": str(stage_manifest_path),
        "model_card": str(model_card_path),
        "evaluation": str(evaluation_path),
        "stage": manifest["stage"],
        "status": manifest["status"],
        "files": sorted(path.name for path in artifact_dir.iterdir() if path.is_file()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and publish the final GRPO LoRA")
    parser.add_argument("--artifact-dir", default="outputs/v2/grpo/final")
    parser.add_argument("--stage-manifest", default="outputs/v2/grpo/stage_manifest.json")
    parser.add_argument("--model-card", default="MODEL_CARD.md")
    parser.add_argument("--evaluation", default="outputs/v2/stage_matrix/stage_matrix.json")
    parser.add_argument(
        "--repo-id",
        default="jatshi/Audio-Codec-LLM-Qwen2.5-1.5B-GRPO-LoRA",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    stage_manifest_path = Path(args.stage_manifest).resolve()
    model_card_path = Path(args.model_card).resolve()
    evaluation_path = Path(args.evaluation).resolve()
    summary = validate_release(
        artifact_dir,
        stage_manifest_path,
        model_card_path,
        evaluation_path,
    )
    summary.update({"repo_id": args.repo_id, "private": args.private})
    if args.dry_run:
        summary["publish_status"] = "dry_run_validated"
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("set HF_TOKEN in the environment; never pass it on the command line")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=artifact_dir,
        commit_message="Upload verified Audio-Codec-LLM GRPO LoRA",
        ignore_patterns=["README.md", "checkpoint-*/*", "optimizer.pt", "scheduler.pt"],
    )
    api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=stage_manifest_path,
        path_in_repo="run_manifest.json",
        commit_message="Add verified GRPO run manifest",
    )
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=evaluation_path.parent,
        path_in_repo="evaluation",
        commit_message="Add audited four-stage holdout predictions and evaluation",
    )
    result = api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        commit_message="Publish audited model card and final evaluation",
    )
    summary["publish_status"] = "uploaded"
    summary["commit_url"] = str(result)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
