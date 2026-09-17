"""Bootstrap time-local EnhanceScript plans from existing global prescriptions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from lse_v2.io import read_jsonl, write_jsonl

from .contracts import ENHANCE_SCRIPT_SCHEMA, validate_enhance_script


def bootstrap_enhance_script(prescription: dict[str, Any], *, duration_ms: int) -> dict[str, Any]:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    diagnosis = str(prescription.get("diagnosis", {}).get("noise_type", "unknown"))
    actions = prescription.get("actions", [])
    action = actions[0] if actions and isinstance(actions[0], dict) else {}
    action_type = str(action.get("type", "hold"))
    reduction = action.get("reduction_db", action.get("gain_db", 0.0))
    strength = min(1.0, max(0.0, abs(float(reduction)) / 24.0))
    script = {
        "schema_version": ENHANCE_SCRIPT_SCHEMA,
        "duration_ms": int(duration_ms),
        "preserve": {
            "speech_content": True,
            "speaker_identity": True,
            "prosody": True,
        },
        "segments": [
            {
                "start_ms": 0,
                "end_ms": int(duration_ms),
                "diagnosis": diagnosis,
                "action": action_type,
                "strength": strength,
                "confidence": float(prescription.get("confidence", 0.0)),
            }
        ],
    }
    validate_enhance_script(script)
    return script


def augment_manifest(source: Path, output: Path, *, duration_ms: int) -> dict[str, Any]:
    rows = read_jsonl(source)
    augmented = []
    prescriptions = 0
    for row in rows:
        item = copy.deepcopy(row)
        for key in ("oracle_prescription", "predicted_prescription"):
            prescription = item.get(key)
            if isinstance(prescription, dict) and "enhance_script" not in prescription:
                prescription["enhance_script"] = bootstrap_enhance_script(
                    prescription, duration_ms=duration_ms
                )
                prescriptions += 1
        augmented.append(item)
    write_jsonl(output, augmented)
    return {
        "schema_version": "lse.enhance_script_augmentation.v1",
        "records": len(augmented),
        "prescriptions_augmented": prescriptions,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=2000)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            augment_manifest(args.input, args.output, duration_ms=args.duration_ms),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
