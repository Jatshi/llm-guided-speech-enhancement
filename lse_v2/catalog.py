"""Build license-aware clean/noise/RIR catalogs from user-provided audio folders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .io import utc_now, write_json_atomic, write_jsonl

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_audio_catalog(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    license_name: str,
    role: str,
    noise_type: str | None = None,
    language: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(input_dir).expanduser().resolve()
    if role not in {"clean", "noise", "rir"}:
        raise ValueError("role must be clean, noise, or rir")
    if not root.is_dir() or not dataset.strip() or not license_name.strip():
        raise ValueError("input directory, dataset, and license are required")
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    if not paths:
        raise ValueError(f"no supported audio files found below {root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root)
        source_id = relative.with_suffix("").as_posix()
        speaker_id = relative.parts[0] if len(relative.parts) > 1 else source_id
        row = {
            "audio_path": str(path),
            "source_id": source_id,
            "speaker_id": speaker_id,
            "dataset": dataset,
            "license": license_name,
            "role": role,
            "file_sha256": _file_sha256(path),
        }
        if role == "noise":
            row["noise_type"] = noise_type or relative.parent.name or "unknown"
        if language:
            row["language"] = language
        if device:
            row["device"] = device
        rows.append(row)
    output = Path(output_path).expanduser().resolve()
    write_jsonl(output, rows)
    report = {
        "schema_version": "lse.audio_catalog.v1",
        "created_at": utc_now(),
        "root": str(root),
        "dataset": dataset,
        "license": license_name,
        "role": role,
        "records": len(rows),
        "catalog": str(output),
    }
    write_json_atomic(output.with_suffix(".report.json"), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--license", dest="license_name", required=True)
    parser.add_argument("--role", choices=("clean", "noise", "rir"), required=True)
    parser.add_argument("--noise-type")
    parser.add_argument("--language")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    report = build_audio_catalog(
        args.input_dir,
        args.output,
        dataset=args.dataset,
        license_name=args.license_name,
        role=args.role,
        noise_type=args.noise_type,
        language=args.language,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
