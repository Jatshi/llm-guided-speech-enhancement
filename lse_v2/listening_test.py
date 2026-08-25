"""Create and aggregate reproducible double-blind A/B listening tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from .io import read_jsonl, utc_now, write_json_atomic, write_jsonl

SCORE_DIMENSIONS = ("naturalness", "intelligibility", "noise_residual", "musical_noise")


def build_blind_listening_pack(
    sample_results: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    max_trials: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(sample_results)
    if max_trials is not None:
        rows = rows[:max_trials]
    if not rows:
        raise ValueError("sample results are empty")
    target = Path(output_dir).expanduser().resolve()
    audio_dir = target / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    trials: list[dict[str, Any]] = []
    key: dict[str, dict[str, Any]] = {}
    for row in rows:
        audit = json.loads(Path(row["audit_report"]).read_text(encoding="utf-8"))
        source = Path(audit["source_audio"])
        candidate = Path(audit["artifacts"]["candidate_audio"])
        if not source.is_file() or not candidate.is_file():
            raise FileNotFoundError(source if not source.is_file() else candidate)
        trial_id = "trial-" + hashlib.sha256(row["sample_id"].encode()).hexdigest()[:16]
        candidate_side = rng.choice(("A", "B"))
        sides = {
            candidate_side: candidate,
            "B" if candidate_side == "A" else "A": source,
        }
        copied: dict[str, str] = {}
        for side, source_path in sides.items():
            output = audio_dir / f"{trial_id}_{side}{source_path.suffix.lower()}"
            shutil.copy2(source_path, output)
            copied[side] = str(output)
        trials.append(
            {
                "trial_id": trial_id,
                "audio_a": copied["A"],
                "audio_b": copied["B"],
                "required_response_fields": [
                    "listener_id",
                    "preference:A|B|tie",
                    *[
                        f"{side.lower()}_{dimension}:1..5"
                        for side in ("A", "B")
                        for dimension in SCORE_DIMENSIONS
                    ],
                ],
            }
        )
        key[trial_id] = {
            "sample_id": row["sample_id"],
            "candidate_side": candidate_side,
            "source_side": "B" if candidate_side == "A" else "A",
        }
    write_jsonl(target / "trials.jsonl", trials)
    write_json_atomic(target / "blind_key.json", key)
    report = {
        "schema_version": "lse.listening_pack.v1",
        "created_at": utc_now(),
        "seed": seed,
        "trials": len(trials),
        "trial_manifest": str(target / "trials.jsonl"),
        "blind_key": str(target / "blind_key.json"),
        "warning": "Keep blind_key.json hidden from listeners until annotation is complete.",
    }
    write_json_atomic(target / "pack_manifest.json", report)
    return report


def aggregate_listening_responses(
    responses: list[dict[str, Any]], key: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not responses:
        raise ValueError("listening responses are empty")
    candidate_wins = source_wins = ties = 0
    listeners: set[str] = set()
    dimension_scores: dict[str, dict[str, list[float]]] = {
        name: {"candidate": [], "source": []} for name in SCORE_DIMENSIONS
    }
    for response in responses:
        trial_id = response.get("trial_id")
        listener = response.get("listener_id")
        preference = response.get("preference")
        if trial_id not in key or not isinstance(listener, str) or not listener:
            raise ValueError("each response requires a known trial_id and listener_id")
        if preference not in {"A", "B", "tie"}:
            raise ValueError("preference must be A, B, or tie")
        listeners.add(listener)
        candidate_side = key[trial_id]["candidate_side"]
        if preference == "tie":
            ties += 1
        elif preference == candidate_side:
            candidate_wins += 1
        else:
            source_wins += 1
        source_side = "B" if candidate_side == "A" else "A"
        for dimension in SCORE_DIMENSIONS:
            for role, side in (("candidate", candidate_side), ("source", source_side)):
                value = response.get(f"{side.lower()}_{dimension}")
                if not isinstance(value, int | float) or not 1 <= value <= 5:
                    # Dimensions may be omitted during a quick pairwise-only test.
                    if value is None:
                        continue
                    raise ValueError(f"{side.lower()}_{dimension} must be in [1, 5]")
                dimension_scores[dimension][role].append(float(value))
    count = len(responses)
    report: dict[str, Any] = {
        "schema_version": "lse.listening_report.v1",
        "responses": count,
        "listeners": len(listeners),
        "candidate_preferred_rate": candidate_wins / count,
        "source_preferred_rate": source_wins / count,
        "tie_rate": ties / count,
    }
    for dimension, scores in dimension_scores.items():
        report[dimension] = {
            "candidate_mean": mean(scores["candidate"]) if scores["candidate"] else None,
            "source_mean": mean(scores["source"]) if scores["source"] else None,
            "rated_responses": min(len(scores["candidate"]), len(scores["source"])),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--sample-results", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--max-trials", type=int)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--responses", type=Path, required=True)
    aggregate.add_argument("--blind-key", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_blind_listening_pack(
            args.sample_results, args.output_dir, seed=args.seed, max_trials=args.max_trials
        )
    else:
        responses = read_jsonl(args.responses)
        key = json.loads(args.blind_key.read_text(encoding="utf-8"))
        report = aggregate_listening_responses(responses, key)
        report["created_at"] = utc_now()
        write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
