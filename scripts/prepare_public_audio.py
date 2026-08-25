#!/usr/bin/env python3
"""Convert public LibriSpeech/DEMAND/SLR28 downloads into licensed LSE catalogs."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

from lse_v2.catalog import build_audio_catalog


def _safe_member_path(root: Path, member: str) -> Path:
    parts = [part for part in PurePosixPath(member).parts if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError(f"unsafe archive member: {member}")
    target = root.joinpath(*parts).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"archive member escapes output root: {member}")
    return target


def _extract_selected(archive: Path, output: Path, predicate) -> int:
    count = 0
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir() or not predicate(info.filename):
                continue
            target = _safe_member_path(output, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != info.file_size:
                with handle.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
            count += 1
    return count


def export_librispeech(parquet_root: Path, output: Path, limit: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    exported = 0
    for parquet_path in sorted(parquet_root.glob("*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256, columns=["audio", "speaker_id", "id"]):
            for row in batch.to_pylist():
                audio = row["audio"]
                payload = audio.get("bytes") if isinstance(audio, dict) else None
                if not payload:
                    raise ValueError(f"embedded audio missing for {row['id']} in {parquet_path}")
                suffix = Path(str(audio.get("path") or "audio.flac")).suffix or ".flac"
                target = output / str(row["speaker_id"]) / f"{row['id']}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != len(payload):
                    target.write_bytes(payload)
                exported += 1
                if exported >= limit:
                    return exported
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--demand-dir", type=Path)
    parser.add_argument("--esc50-zip", type=Path)
    parser.add_argument("--rir-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-clean", type=int, default=10_000)
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    clean_root, noise_root, rir_root = root / "clean", root / "noise", root / "rir"
    clean_count = export_librispeech(args.parquet_root, clean_root, args.max_clean)
    if clean_count < args.max_clean:
        raise RuntimeError(f"requested {args.max_clean} clean files, found {clean_count}")

    if args.demand_dir is None and args.esc50_zip is None:
        raise ValueError("provide --demand-dir, --esc50-zip, or both")
    noise_count = 0
    noise_datasets: list[str] = []
    noise_licenses: list[str] = []
    if args.demand_dir is not None:
        for archive in sorted(args.demand_dir.glob("*_16k.zip")):
            destination = noise_root / "DEMAND" / archive.stem
            noise_count += _extract_selected(
                archive, destination, lambda name: Path(name).suffix.lower() == ".wav"
            )
        noise_datasets.append("DEMAND 16 kHz selected environments")
        noise_licenses.append("CC BY-SA 3.0")
    if args.esc50_zip is not None:
        noise_count += _extract_selected(
            args.esc50_zip,
            noise_root / "ESC-50",
            lambda name: "/audio/" in name.lower() and Path(name).suffix.lower() == ".wav",
        )
        noise_datasets.append("ESC-50 environmental recordings")
        noise_licenses.append("CC BY-NC 3.0")
    if noise_count == 0:
        raise RuntimeError("no DEMAND WAV files were extracted")

    rir_count = _extract_selected(
        args.rir_zip,
        rir_root,
        lambda name: "simulated_rirs" in name.lower() and Path(name).suffix.lower() == ".wav",
    )
    if rir_count == 0:
        raise RuntimeError("no SLR28 simulated RIR WAV files were extracted")

    catalogs = root / "catalogs"
    catalogs.mkdir(parents=True, exist_ok=True)
    clean_report = build_audio_catalog(
        clean_root,
        catalogs / "clean.jsonl",
        dataset="LibriSpeech train-clean-100 (ModelScope mirror)",
        license_name="CC BY 4.0",
        role="clean",
        language="en",
    )
    noise_report = build_audio_catalog(
        noise_root,
        catalogs / "noise.jsonl",
        dataset=" + ".join(noise_datasets),
        license_name=" + ".join(noise_licenses),
        role="noise",
    )
    rir_report = build_audio_catalog(
        rir_root,
        catalogs / "rir.jsonl",
        dataset="OpenSLR SLR28 simulated RIRs",
        license_name="Apache-2.0",
        role="rir",
    )
    provenance = {
        "schema_version": "lse.public_audio_sources.v1",
        "sources": [
            {
                "dataset": "LibriSpeech",
                "url": "https://www.modelscope.cn/datasets/openslr/librispeech_asr",
                "subset": "train.clean.100",
                "license": "CC BY 4.0",
            },
            *(
                [
                    {
                        "dataset": "DEMAND",
                        "url": "https://zenodo.org/records/1227121",
                        "license": "CC BY-SA 3.0",
                    }
                ]
                if args.demand_dir is not None
                else []
            ),
            *(
                [
                    {
                        "dataset": "ESC-50",
                        "url": "https://www.modelscope.cn/datasets/OmniData/ESC-50",
                        "upstream": "https://github.com/karoldvl/ESC-50",
                        "license": "CC BY-NC 3.0",
                    }
                ]
                if args.esc50_zip is not None
                else []
            ),
            {
                "dataset": "SLR28 RIR and Noise Database",
                "url": "https://www.openslr.org/28/",
                "subset": "simulated_rirs",
                "license": "Apache-2.0",
            },
        ],
        "catalogs": {"clean": clean_report, "noise": noise_report, "rir": rir_report},
    }
    (root / "public_source_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
