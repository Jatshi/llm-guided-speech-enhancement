"""Versioned paired-audio and structured-prescription contracts for MM-DiT."""

from __future__ import annotations

import hashlib
import math
from typing import Any

MMDIT_PAIR_SCHEMA = "lse.mmdit_pair.v1"
ENHANCE_SCRIPT_SCHEMA = "lse.enhance_script.v1"
VALID_SPLITS = {"train", "validation", "test"}


class PairContractError(ValueError):
    """Raised when a paired enhancement record is incomplete or unsafe."""


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PairContractError(f"{name} must be a non-empty string")
    return value


def validate_prescription(value: Any, context: str = "prescription") -> None:
    if not isinstance(value, dict):
        raise PairContractError(f"{context} must be an object")
    diagnosis = value.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise PairContractError(f"{context}.diagnosis must be an object")
    actions = value.get("actions")
    if not isinstance(actions, list):
        raise PairContractError(f"{context}.actions must be a list")
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise PairContractError(f"{context}.actions[{index}] must be an object")
        _non_empty_string(action.get("type"), f"{context}.actions[{index}].type")
        for key, item in action.items():
            if key == "type":
                continue
            if isinstance(item, int | float) and not math.isfinite(float(item)):
                raise PairContractError(f"{context}.actions[{index}].{key} is not finite")
    confidence = value.get("confidence", 0.0)
    if not isinstance(confidence, int | float) or not 0 <= float(confidence) <= 1:
        raise PairContractError(f"{context}.confidence must be in [0, 1]")
    script = value.get("enhance_script")
    if script is not None:
        validate_enhance_script(script, f"{context}.enhance_script")


def validate_enhance_script(value: Any, context: str = "enhance_script") -> None:
    """Validate a readable, time-local enhancement plan before tokenization."""

    if not isinstance(value, dict):
        raise PairContractError(f"{context} must be an object")
    if value.get("schema_version") != ENHANCE_SCRIPT_SCHEMA:
        raise PairContractError(f"{context}.schema_version must be {ENHANCE_SCRIPT_SCHEMA}")
    duration = value.get("duration_ms")
    if not isinstance(duration, int | float) or not math.isfinite(float(duration)) or duration <= 0:
        raise PairContractError(f"{context}.duration_ms must be positive and finite")
    preserve = value.get("preserve")
    if not isinstance(preserve, dict):
        raise PairContractError(f"{context}.preserve must be an object")
    for key in ("speech_content", "speaker_identity", "prosody"):
        if not isinstance(preserve.get(key), bool):
            raise PairContractError(f"{context}.preserve.{key} must be boolean")
    segments = value.get("segments")
    if not isinstance(segments, list) or not segments:
        raise PairContractError(f"{context}.segments must be a non-empty list")
    previous_end = 0.0
    for index, segment in enumerate(segments):
        prefix = f"{context}.segments[{index}]"
        if not isinstance(segment, dict):
            raise PairContractError(f"{prefix} must be an object")
        start = segment.get("start_ms")
        end = segment.get("end_ms")
        if not all(
            isinstance(item, int | float) and math.isfinite(float(item)) for item in (start, end)
        ):
            raise PairContractError(f"{prefix} boundaries must be finite numbers")
        start_value, end_value = float(start), float(end)
        if start_value < previous_end:
            raise PairContractError(f"{prefix} overlaps the previous segment")
        if start_value < 0 or end_value <= start_value or end_value > float(duration):
            raise PairContractError(
                f"{prefix} boundaries must satisfy 0 <= start < end <= duration"
            )
        _non_empty_string(segment.get("diagnosis"), f"{prefix}.diagnosis")
        _non_empty_string(segment.get("action"), f"{prefix}.action")
        strength = segment.get("strength")
        confidence = segment.get("confidence", 0.0)
        if not isinstance(strength, int | float) or not 0 <= float(strength) <= 1:
            raise PairContractError(f"{prefix}.strength must be in [0, 1]")
        if not isinstance(confidence, int | float) or not 0 <= float(confidence) <= 1:
            raise PairContractError(f"{prefix}.confidence must be in [0, 1]")
        previous_end = end_value


def validate_pair_record(record: dict[str, Any], *, check_files: bool = False) -> None:
    from pathlib import Path

    context = _non_empty_string(record.get("sample_id"), "sample_id")
    if record.get("schema_version") != MMDIT_PAIR_SCHEMA:
        raise PairContractError(f"{context}: schema_version must be {MMDIT_PAIR_SCHEMA}")
    _non_empty_string(record.get("speaker_id"), f"{context}.speaker_id")
    if record.get("split") not in VALID_SPLITS:
        raise PairContractError(f"{context}.split must be train/validation/test")
    audio = record.get("audio")
    if not isinstance(audio, dict):
        raise PairContractError(f"{context}.audio must be an object")
    noisy = _non_empty_string(audio.get("noisy_path"), f"{context}.audio.noisy_path")
    clean = _non_empty_string(audio.get("clean_path"), f"{context}.audio.clean_path")
    sample_rate = audio.get("sample_rate")
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 48000:
        raise PairContractError(f"{context}.audio.sample_rate must be in [8000, 48000]")
    if Path(noisy).resolve() == Path(clean).resolve():
        raise PairContractError(f"{context}: noisy_path and clean_path must differ")
    if check_files:
        for name, path in (("noisy_path", noisy), ("clean_path", clean)):
            if not Path(path).expanduser().is_file():
                raise PairContractError(f"{context}.audio.{name} not found: {path}")
    validate_prescription(record.get("oracle_prescription"), f"{context}.oracle_prescription")
    predicted = record.get("predicted_prescription")
    if predicted is not None:
        validate_prescription(predicted, f"{context}.predicted_prescription")
    if not isinstance(record.get("provenance", {}), dict):
        raise PairContractError(f"{context}.provenance must be an object")


def _category(value: Any, buckets: int = 1024) -> int:
    if value in (None, "", False):
        return 1 if value is False else 0
    digest = hashlib.sha256(str(value).strip().lower().encode()).digest()
    return 2 + int.from_bytes(digest[:4], "big") % (buckets - 2)


def _finite_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return default


def prescription_tokens(
    prescription: dict[str, Any] | None, *, max_tokens: int = 8
) -> list[tuple[int, int, float]]:
    """Convert JSON into stable ``(field, category, value)`` tokens.

    This intentionally retains an auditable mapping rather than hiding the planner
    output inside an opaque sentence embedding.
    """
    if max_tokens < 4:
        raise ValueError("max_tokens must be at least 4")
    if prescription is None:
        return [(0, 0, 0.0)] * max_tokens
    validate_prescription(prescription)
    diagnosis = prescription.get("diagnosis", {})
    result = [
        (1, _category(diagnosis.get("noise_type")), 0.0),
        (2, _category(bool(diagnosis.get("reverb"))), float(bool(diagnosis.get("reverb")))),
        (
            3,
            _category(bool(diagnosis.get("band_limited"))),
            float(bool(diagnosis.get("band_limited"))),
        ),
        (4, 1, _finite_number(prescription.get("confidence"), 0.0)),
    ]
    for index, action in enumerate(prescription.get("actions", [])):
        numeric = next(
            (
                _finite_number(action[key])
                for key in ("reduction_db", "gain_db", "q")
                if key in action
            ),
            0.0,
        )
        result.append(
            (10 + index, _category(action.get("type")), max(-1.0, min(1.0, numeric / 40)))
        )
        if len(result) == max_tokens:
            break
    script = prescription.get("enhance_script")
    if script is not None and len(result) < max_tokens:
        scripted = enhance_script_tokens(script, max_tokens=max_tokens)
        result.extend(token for token in scripted if token[0] != 0)
    result.extend([(0, 0, 0.0)] * (max_tokens - len(result)))
    return result[:max_tokens]


def enhance_script_tokens(
    script: dict[str, Any], *, max_tokens: int = 16
) -> list[tuple[int, int, float]]:
    """Encode global preservation constraints and local timeline decisions.

    Each segment receives three tokens: normalized start, normalized end, and
    executable action strength.  Diagnosis is retained as the category on both
    boundary tokens so truncation never leaves an unlabeled time interval.
    """

    if max_tokens < 4:
        raise ValueError("max_tokens must be at least 4")
    validate_enhance_script(script)
    preserve = script["preserve"]
    duration = float(script["duration_ms"])
    result: list[tuple[int, int, float]] = [
        (5, 1, float(preserve["speech_content"])),
        (6, 1, float(preserve["speaker_identity"])),
        (7, 1, float(preserve["prosody"])),
    ]
    for index, segment in enumerate(script["segments"]):
        field = 20 + 3 * index
        diagnosis = _category(segment["diagnosis"])
        result.extend(
            [
                (field, diagnosis, float(segment["start_ms"]) / duration),
                (field + 1, diagnosis, float(segment["end_ms"]) / duration),
                (field + 2, _category(segment["action"]), float(segment["strength"])),
            ]
        )
        if len(result) >= max_tokens:
            break
    result.extend([(0, 0, 0.0)] * (max_tokens - len(result)))
    return result[:max_tokens]
