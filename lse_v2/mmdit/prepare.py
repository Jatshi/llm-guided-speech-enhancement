"""Normalize existing LSE manifests into paired MM-DiT records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lse_v2.io import read_jsonl, write_jsonl

from .contracts import MMDIT_PAIR_SCHEMA, PairContractError, validate_pair_record


def _parse_json(value: Any, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PairContractError(f"{context} is not valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
    raise PairContractError(f"{context} must resolve to an object")


def _speaker_id(record: dict[str, Any]) -> str:
    explicit = record.get("speaker_id") or record.get("provenance", {}).get("speaker_id")
    if explicit:
        return str(explicit)
    sample_id = str(record.get("sample_id", "unknown"))
    aishell = re.search(r"(?:^|[-_])(S\d{4,})(?:[-_]|$)", sample_id, re.IGNORECASE)
    if aishell:
        return aishell.group(1).upper()
    return sample_id.split("-", 1)[0]


def _split(value: Any) -> str:
    mapping = {"eval": "validation", "val": "validation"}
    normalized = mapping.get(str(value).lower(), str(value).lower())
    return normalized if normalized in {"train", "validation", "test"} else "train"


def _convert(record: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(record.get("sample_id", "")).strip()
    if record.get("schema_version") == "lse.audio_manifest.v2":
        audio = record.get("audio", {})
        noisy = audio.get("noisy_path")
        clean = audio.get("clean_path")
        sample_rate = audio.get("sample_rate", 16000)
        oracle = record.get("target")
    else:
        noisy = record.get("audio_path") or record.get("noisy_path")
        clean = record.get("clean_path")
        sample_rate = record.get("sample_rate", 16000)
        oracle = record.get("oracle_prescription") or record.get("chosen_text")
    predicted = record.get("predicted_prescription") or record.get("prediction")
    result = {
        "schema_version": MMDIT_PAIR_SCHEMA,
        "sample_id": sample_id,
        "speaker_id": _speaker_id(record),
        "split": _split(record.get("split", "train")),
        "audio": {
            "noisy_path": str(noisy or ""),
            "clean_path": str(clean or ""),
            "sample_rate": int(sample_rate),
        },
        "oracle_prescription": _parse_json(oracle, f"{sample_id}.oracle_prescription"),
        "predicted_prescription": _parse_json(predicted, f"{sample_id}.predicted_prescription"),
        "provenance": {
            **(record.get("provenance") if isinstance(record.get("provenance"), dict) else {}),
            "source_schema": record.get("schema_version", "legacy-aligned"),
        },
    }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pairs(
    input_manifest: Path,
    output_manifest: Path,
    *,
    check_files: bool = False,
    require_speaker_disjoint: bool = False,
) -> dict[str, Any]:
    source = read_jsonl(input_manifest)
    if not source:
        raise PairContractError("input manifest is empty")
    converted = [_convert(record) for record in source]
    seen: set[str] = set()
    speakers: dict[str, set[str]] = defaultdict(set)
    for record in converted:
        validate_pair_record(record, check_files=check_files)
        if record["sample_id"] in seen:
            raise PairContractError(f"duplicate sample_id: {record['sample_id']}")
        seen.add(record["sample_id"])
        speakers[record["speaker_id"]].add(record["split"])
    leakage = {speaker: sorted(splits) for speaker, splits in speakers.items() if len(splits) > 1}
    if leakage and require_speaker_disjoint:
        preview = dict(list(leakage.items())[:5])
        raise PairContractError(f"speaker leakage across splits: {preview}")
    write_jsonl(output_manifest, converted)
    report = {
        "schema_version": "lse.mmdit_pair_bundle.v1",
        "source_manifest": str(input_manifest.resolve()),
        "source_sha256": _sha256(input_manifest),
        "output_manifest": str(output_manifest.resolve()),
        "records": len(converted),
        "split_counts": dict(Counter(record["split"] for record in converted)),
        "speakers": len(speakers),
        "speaker_leakage_count": len(leakage),
        "predicted_prescriptions": sum(
            record["predicted_prescription"] is not None for record in converted
        ),
    }
    report_path = output_manifest.with_suffix(".manifest.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--require-speaker-disjoint", action="store_true")
    args = parser.parse_args(argv)
    report = prepare_pairs(
        args.input,
        args.output,
        check_files=args.check_files,
        require_speaker_disjoint=args.require_speaker_disjoint,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
