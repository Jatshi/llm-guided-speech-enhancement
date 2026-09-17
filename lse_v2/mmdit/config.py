"""Configuration validation and path resolution for the MM-DiT experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class MMDiTConfigError(ValueError):
    pass


def load_mmdit_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MMDiTConfigError("top-level config must be an object")
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    validate_mmdit_config(data)
    return data


def validate_mmdit_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "lse.mmdit.config.v1":
        raise MMDiTConfigError("schema_version must be lse.mmdit.config.v1")
    for section in ("project", "data", "codec", "model", "flow", "training", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise MMDiTConfigError(f"missing configuration section: {section}")
    data = config["data"]
    if float(data.get("segment_seconds", 0)) <= 0:
        raise MMDiTConfigError("data.segment_seconds must be positive")
    if int(data.get("sample_rate", 0)) not in {8000, 16000, 24000, 32000, 48000}:
        raise MMDiTConfigError("data.sample_rate must be a supported speech rate")
    training = config["training"]
    for key in ("batch_size", "gradient_accumulation_steps", "max_steps"):
        if int(training.get(key, 0)) < 1:
            raise MMDiTConfigError(f"training.{key} must be >= 1")
    if float(training.get("learning_rate", 0)) <= 0:
        raise MMDiTConfigError("training.learning_rate must be positive")


def project_root(config: dict[str, Any]) -> Path:
    root = Path(os.path.expandvars(str(config["project"].get("root", ".")))).expanduser()
    if not root.is_absolute():
        root = Path(config["_config_dir"]) / root
    return root.resolve()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def config_digest(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()
