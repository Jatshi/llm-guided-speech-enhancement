"""Versioned data contracts for audio evidence and alignment datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl

AUDIO_SCHEMA = "lse.audio_manifest.v2"
SFT_SCHEMA = "lse.sft.v2"
DPO_SCHEMA = "lse.dpo.v2"
GRPO_SCHEMA = "lse.grpo.v2"

SYSTEM_PROMPT = (
    "You are an evidence-grounded speech enhancement planner. "
    "Return exactly one JSON object with keys diagnosis, actions, rationale, and confidence. "
    "Never invent measurements that are absent from the acoustic evidence."
)


class ContractError(ValueError):
    """Raised when a dataset record violates its declared schema."""


def _require(record: dict[str, Any], key: str, expected: type, context: str) -> Any:
    if key not in record:
        raise ContractError(f"{context}: missing field '{key}'")
    value = record[key]
    if not isinstance(value, expected):
        raise ContractError(
            f"{context}.{key}: expected {expected.__name__}, got {type(value).__name__}"
        )
    return value


def validate_audio_record(record: dict[str, Any], check_files: bool = False) -> None:
    context = str(record.get("sample_id", "<unknown>"))
    if record.get("schema_version") != AUDIO_SCHEMA:
        raise ContractError(f"{context}: schema_version must be {AUDIO_SCHEMA}")
    _require(record, "sample_id", str, context)
    audio = _require(record, "audio", dict, context)
    noisy_path = _require(audio, "noisy_path", str, f"{context}.audio")
    sample_rate = _require(audio, "sample_rate", int, f"{context}.audio")
    if sample_rate < 8000 or sample_rate > 192000:
        raise ContractError(f"{context}.audio.sample_rate outside [8000, 192000]")
    duration = audio.get("duration_seconds")
    if duration is not None and (not isinstance(duration, int | float) or duration <= 0):
        raise ContractError(f"{context}.audio.duration_seconds must be positive")
    if check_files and not Path(noisy_path).expanduser().is_file():
        raise ContractError(f"{context}: noisy audio file not found: {noisy_path}")
    acoustics = _require(record, "acoustics", dict, context)
    noise_type = _require(acoustics, "noise_type", str, f"{context}.acoustics")
    if not noise_type.strip():
        raise ContractError(f"{context}.acoustics.noise_type cannot be empty")
    snr = acoustics.get("snr_db")
    if snr is not None and (not isinstance(snr, int | float) or not -30 <= snr <= 80):
        raise ContractError(f"{context}.acoustics.snr_db outside [-30, 80]")
    target = _require(record, "target", dict, context)
    _require(target, "diagnosis", dict, f"{context}.target")
    actions = _require(target, "actions", list, f"{context}.target")
    if not actions:
        raise ContractError(f"{context}.target.actions cannot be empty")
    _require(target, "rationale", str, f"{context}.target")
    split = record.get("split", "train")
    if split not in {"train", "eval", "test", "unassigned"}:
        raise ContractError(f"{context}.split must be train/eval/test/unassigned")


def validate_alignment_record(record: dict[str, Any]) -> None:
    schema = record.get("schema_version")
    context = str(record.get("sample_id", "<unknown>"))
    if schema == SFT_SCHEMA:
        messages = _require(record, "messages", list, context)
        _validate_messages(messages, context, require_assistant=True)
    elif schema == DPO_SCHEMA:
        _validate_messages(_require(record, "prompt", list, context), context, False)
        _validate_messages(_require(record, "chosen", list, context), context, True)
        _validate_messages(_require(record, "rejected", list, context), context, True)
        _require(record, "preference", dict, context)
    elif schema == GRPO_SCHEMA:
        _validate_messages(_require(record, "prompt", list, context), context, False)
        _require(record, "ground_truth", dict, context)
        _require(record, "reward_context", dict, context)
    else:
        raise ContractError(f"{context}: unsupported schema_version {schema!r}")


def _validate_messages(messages: list[Any], context: str, require_assistant: bool) -> None:
    if not messages:
        raise ContractError(f"{context}: messages cannot be empty")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ContractError(f"{context}.messages[{index}] must be an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ContractError(f"{context}.messages[{index}]: invalid role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ContractError(f"{context}.messages[{index}]: content cannot be empty")
    if require_assistant and not any(item["role"] == "assistant" for item in messages):
        raise ContractError(f"{context}: assistant message is required")


def acoustic_evidence(record: dict[str, Any]) -> str:
    audio = record["audio"]
    acoustics = record["acoustics"]
    evidence = {
        "audio_path": audio["noisy_path"],
        "sample_rate": audio["sample_rate"],
        "duration_seconds": audio.get("duration_seconds"),
        "noise_type": acoustics["noise_type"],
        "snr_db": acoustics.get("snr_db"),
        "reverb_rt60": acoustics.get("reverb_rt60"),
        "bandlimit_hz": acoustics.get("bandlimit_hz"),
        "features": acoustics.get("features", {}),
    }
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def target_json(record: dict[str, Any]) -> str:
    target = dict(record["target"])
    target.setdefault("confidence", 0.8)
    return json.dumps(target, ensure_ascii=False, sort_keys=True)


def corrupt_target(target: dict[str, Any]) -> dict[str, Any]:
    """Create a deterministic unsafe preference negative without mutating input."""
    bad = json.loads(json.dumps(target))
    actions = bad.setdefault("actions", [])
    if actions:
        action = actions[0]
        if "reduction_db" in action:
            action["reduction_db"] = 36.0
        elif "gain_db" in action:
            action["gain_db"] = -30.0
        else:
            action["reduction_db"] = 36.0
    else:
        actions.append(
            {
                "type": "spectral_subtraction",
                "reduction_db": 36.0,
                "low_hz": 0,
                "high_hz": 8000,
            }
        )
    bad["rationale"] = "Apply maximum full-band suppression regardless of the measured evidence."
    bad["confidence"] = 0.99
    return bad


def derive_alignment_records(
    audio_records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    grpo: list[dict[str, Any]] = []
    for record in audio_records:
        validate_audio_record(record)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Inspect the following audio/acoustic evidence and produce a safe, "
                    f"executable enhancement prescription.\n{acoustic_evidence(record)}"
                ),
            },
        ]
        chosen_text = target_json(record)
        rejected_text = json.dumps(
            corrupt_target(record["target"]), ensure_ascii=False, sort_keys=True
        )
        common = {
            "sample_id": record["sample_id"],
            "source_audio_schema": AUDIO_SCHEMA,
            "split": record.get("split", "train"),
            "evidence": {
                "audio": record["audio"],
                "acoustics": record["acoustics"],
            },
        }
        sft.append(
            {
                **common,
                "schema_version": SFT_SCHEMA,
                "messages": prompt + [{"role": "assistant", "content": chosen_text}],
            }
        )
        dpo.append(
            {
                **common,
                "schema_version": DPO_SCHEMA,
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
                "preference": {
                    "source": "rule_based_safety_negative",
                    "reason": "chosen is evidence-consistent; rejected over-processes",
                    "margin": 1.0,
                },
            }
        )
        grpo.append(
            {
                **common,
                "schema_version": GRPO_SCHEMA,
                "prompt": prompt,
                "ground_truth": record["target"],
                "reward_context": {
                    "noise_type": record["acoustics"]["noise_type"],
                    "snr_db": record["acoustics"].get("snr_db"),
                    "reverb_rt60": record["acoustics"].get("reverb_rt60"),
                    "bandlimit_hz": record["acoustics"].get("bandlimit_hz"),
                    "expected_response": chosen_text,
                },
            }
        )
    return sft, dpo, grpo


def deterministic_split(
    records: list[dict[str, Any]], seed: int, eval_ratio: float
) -> list[dict[str, Any]]:
    if not 0 <= eval_ratio < 1:
        raise ContractError("eval_ratio must be in [0, 1)")
    result = [json.loads(json.dumps(record)) for record in records]
    unassigned = [
        record for record in result if record.get("split") not in {"train", "eval", "test"}
    ]
    unassigned.sort(
        key=lambda record: hashlib.sha256(f"{seed}:{record['sample_id']}".encode()).digest()
    )
    existing_eval = sum(record.get("split") == "eval" for record in result)
    target_eval = round(len(result) * eval_ratio)
    if eval_ratio > 0 and len(result) >= 2:
        target_eval = max(1, target_eval)
    assign_eval = min(len(unassigned), max(0, target_eval - existing_eval))
    for index, record in enumerate(unassigned):
        record["split"] = "eval" if index < assign_eval else "train"
    return result


def build_alignment_datasets(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    eval_ratio: float = 0.05,
    check_audio_files: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(manifest_path)
    if not records:
        raise ContractError("Audio manifest is empty")
    ids: set[str] = set()
    for record in records:
        validate_audio_record(record, check_files=check_audio_files)
        sample_id = record["sample_id"]
        if sample_id in ids:
            raise ContractError(f"Duplicate sample_id: {sample_id}")
        ids.add(sample_id)
    records = deterministic_split(records, seed, eval_ratio)
    sft, dpo, grpo = derive_alignment_records(records)
    target = Path(output_dir)
    counts: dict[str, dict[str, int]] = {}
    for name, dataset in (("sft", sft), ("dpo", dpo), ("grpo", grpo)):
        counts[name] = {}
        for split in ("train", "eval", "test"):
            subset = [row for row in dataset if row.get("split") == split]
            if subset:
                write_jsonl(target / name / f"{split}.jsonl", subset)
            counts[name][split] = len(subset)
    manifest = {
        "schema_version": "lse.alignment_bundle.v2",
        "source_manifest": str(Path(manifest_path).resolve()),
        "seed": seed,
        "eval_ratio": eval_ratio,
        "counts": counts,
    }
    (target / "dataset_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (target / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def extract_audio_features(path: str | Path, sample_rate: int = 16000) -> dict[str, float]:
    """Extract lightweight, deterministic features used as direct acoustic evidence."""
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the 'audio' extra to extract audio features") from exc
    signal, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    if signal.size == 0:
        raise ContractError(f"Audio is empty: {path}")
    rms = float(np.sqrt(np.mean(np.square(signal))))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=signal)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(signal)))
    duration = float(signal.size / sr)
    values = {
        "rms": rms,
        "spectral_flatness": flatness,
        "spectral_centroid_hz": centroid,
        "zero_crossing_rate": zcr,
        "duration_seconds": duration,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ContractError(f"Non-finite feature extracted from {path}")
    return {key: round(value, 6) for key, value in values.items()}


def manifest_from_legacy(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 16000,
    extract_features: bool = False,
) -> int:
    source = Path(metadata_path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ContractError("Legacy metadata must be a JSON array")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        cfg = item.get("degradation_config", {})
        direct_audio_available = bool(item.get("audio_path"))
        noisy_path = item.get("audio_path") or item.get("clean_path")
        if not noisy_path:
            raise ContractError(f"Legacy record {index} has no audio_path or clean_path")
        features = (
            extract_audio_features(noisy_path, sample_rate)
            if extract_features and direct_audio_available
            else {}
        )
        noise_type = str(cfg.get("noise_type", "unknown"))
        reduction = 8.0 if float(cfg.get("snr_db", 15)) >= 10 else 12.0
        source_provenance = item.get("provenance")
        if not isinstance(source_provenance, dict):
            source_provenance = {}
        record = {
            "schema_version": AUDIO_SCHEMA,
            "sample_id": str(item.get("id", f"legacy-{index:06d}")),
            "split": item.get("split", "unassigned"),
            "audio": {
                "noisy_path": str(noisy_path),
                "clean_path": item.get("clean_path"),
                "source_role": (
                    "materialized_noisy_audio"
                    if direct_audio_available
                    else "clean_proxy_for_synthetic_degradation"
                ),
                "sample_rate": sample_rate,
                "duration_seconds": features.pop("duration_seconds", None),
            },
            "acoustics": {
                "noise_type": noise_type,
                "snr_db": cfg.get("snr_db"),
                "reverb_rt60": cfg.get("reverb_rt60"),
                "bandlimit_hz": cfg.get("bandlimit"),
                "features": features,
            },
            "target": {
                "diagnosis": {
                    "noise_type": noise_type,
                    "reverb": bool(cfg.get("reverb_rt60")),
                    "band_limited": bool(cfg.get("bandlimit")),
                },
                "actions": [
                    {
                        "type": "spectral_subtraction",
                        "reduction_db": reduction,
                        "low_hz": 80,
                        "high_hz": 7600,
                    }
                ],
                "rationale": "Apply conservative suppression derived from measured degradation.",
                "confidence": 0.75,
            },
            "provenance": {
                **source_provenance,
                "dataset": source_provenance.get("dataset", "legacy-import"),
                "source_metadata": str(source.resolve()),
                "direct_noisy_audio_available": direct_audio_available,
            },
        }
        validate_audio_record(record)
        records.append(record)
    write_jsonl(output_path, records)
    return len(records)
