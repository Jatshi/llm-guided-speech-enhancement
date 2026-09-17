"""Versioned paired-audio and structured-prescription contracts for MM-DiT."""

from __future__ import annotations

import hashlib
import math
from typing import Any

MMDIT_PAIR_SCHEMA = "lse.mmdit_pair.v1"
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
    result.extend([(0, 0, 0.0)] * (max_tokens - len(result)))
    return result[:max_tokens]
