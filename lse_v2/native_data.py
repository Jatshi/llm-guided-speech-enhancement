"""Convert physical audio manifests into native-audio alignment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contracts import SYSTEM_PROMPT, corrupt_target, validate_audio_record
from .dsp import plan_from_prescription
from .io import read_jsonl, utc_now, write_json_atomic, write_jsonl
from .native_pipeline import NATIVE_AUDIO_SCHEMA


def _prompt(record: dict[str, Any]) -> str:
    acoustics = record["acoustics"]
    evidence = {
        "measured_features": acoustics.get("features", {}),
        "sample_rate": record["audio"].get("sample_rate"),
        "duration_seconds": record["audio"].get("duration_seconds"),
        "privileged_degradation_labels_in_prompt": False,
    }
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Listen to the attached noisy waveform. The following measurements are auxiliary "
        "evidence and can be missing or imperfect. Diagnose the signal and return a safe "
        "executable prescription.\n"
        f"AUXILIARY_EVIDENCE={json.dumps(evidence, ensure_ascii=False, sort_keys=True)}"
    )


def build_native_manifest(
    audio_manifest: str | Path,
    output_path: str | Path,
    *,
    require_materialized: bool = True,
    preference_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Build the one-record-per-waveform manifest consumed by native training."""

    source_path = Path(audio_manifest).expanduser().resolve()
    rows = read_jsonl(source_path)
    if not rows:
        raise ValueError("audio manifest is empty")
    preferences: dict[str, dict[str, Any]] = {}
    if preference_manifest is not None:
        for preference in read_jsonl(preference_manifest):
            sample_id = preference.get("sample_id")
            required = ("chosen_text", "rejected_text", "annotator_id", "rationale")
            if not isinstance(sample_id, str) or any(
                not isinstance(preference.get(key), str) or not preference[key] for key in required
            ):
                raise ValueError(
                    "human preference rows require sample_id and non-empty annotation fields"
                )
            if sample_id in preferences:
                raise ValueError(f"duplicate human preference for {sample_id}")
            preferences[sample_id] = preference
    native: list[dict[str, Any]] = []
    sources: set[str] = set()
    for row in rows:
        validate_audio_record(row)
        audio = row["audio"]
        if require_materialized and audio.get("source_role") != "materialized_noisy_audio":
            raise ValueError(f"{row['sample_id']} is not materialized noisy audio")
        audio_path = Path(audio["noisy_path"]).expanduser()
        if not audio_path.is_absolute():
            audio_path = (source_path.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        provenance = row.get("provenance") or {}
        source_id = str(
            provenance.get("speaker_id") or provenance.get("source_id") or row["sample_id"]
        )
        sources.add(source_id)
        target = dict(row["target"])
        target.setdefault("confidence", 0.8)
        human = preferences.get(row["sample_id"])
        chosen_text = (
            human["chosen_text"]
            if human
            else json.dumps(target, ensure_ascii=False, sort_keys=True)
        )
        rejected_text = (
            human["rejected_text"]
            if human
            else json.dumps(corrupt_target(target), ensure_ascii=False, sort_keys=True)
        )
        plan_from_prescription(chosen_text)
        rejected_payload = json.loads(rejected_text)
        if not isinstance(rejected_payload, dict):
            raise ValueError(f"rejected preference for {row['sample_id']} must be a JSON object")
        native.append(
            {
                "schema_version": NATIVE_AUDIO_SCHEMA,
                "sample_id": row["sample_id"],
                "source_id": source_id,
                "split": row.get("split", "train"),
                "audio_path": str(audio_path),
                "clean_path": str(Path(audio["clean_path"]).expanduser().resolve())
                if audio.get("clean_path")
                else None,
                "prompt_text": _prompt(row),
                "chosen_text": chosen_text,
                "rejected_text": rejected_text,
                "preference": {
                    "source": "human_annotation" if human else "rule_based_safety_negative",
                    "annotator_id": human.get("annotator_id") if human else None,
                    "rationale": (
                        human["rationale"]
                        if human
                        else "evidence-consistent target beats unsafe over-processing"
                    ),
                },
                "reward_context": {
                    "noise_type": row["acoustics"].get("noise_type"),
                    "snr_db": row["acoustics"].get("snr_db"),
                    "reverb_rt60": row["acoustics"].get("reverb_rt60"),
                    "bandlimit_hz": row["acoustics"].get("bandlimit_hz"),
                    "expected_response": json.dumps(target, ensure_ascii=False, sort_keys=True),
                },
                "provenance": provenance,
            }
        )
    output = Path(output_path).expanduser().resolve()
    write_jsonl(output, native)
    split_counts = {
        split: sum(row["split"] == split for row in native) for split in ("train", "eval", "test")
    }
    report = {
        "schema_version": "lse.native_manifest_build.v1",
        "created_at": utc_now(),
        "source_manifest": str(source_path),
        "output_manifest": str(output),
        "records": len(native),
        "sources": len(sources),
        "split_counts": split_counts,
        "all_audio_materialized": True,
        "human_preference_records": sum(
            row["preference"]["source"] == "human_annotation" for row in native
        ),
    }
    write_json_atomic(output.with_suffix(".report.json"), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preference-manifest", type=Path)
    args = parser.parse_args(argv)
    report = build_native_manifest(
        args.audio_manifest,
        args.output,
        preference_manifest=args.preference_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
