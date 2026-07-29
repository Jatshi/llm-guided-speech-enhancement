"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a training configuration is incomplete or unsafe."""


REQUIRED_STAGES = ("sft", "dpo", "grpo")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config does not exist: {config_path}")
    if config_path.suffix.lower() == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("PyYAML is required for YAML configuration files") from exc
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ConfigError("Configuration must be JSON or YAML")
    if not isinstance(data, dict):
        raise ConfigError("Top-level configuration must be an object")
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    for key in ("project", "model", "data", "training"):
        if key not in config:
            raise ConfigError(f"Missing required configuration section: {key}")
    model_id = config["model"].get("name_or_path")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("model.name_or_path must be a non-empty string")
    stages = config["training"].get("stages", {})
    for stage in REQUIRED_STAGES:
        if stage not in stages:
            raise ConfigError(f"Missing training.stages.{stage}")
        batch_size = int(stages[stage].get("per_device_train_batch_size", 0))
        if batch_size < 1:
            raise ConfigError(f"{stage} batch size must be >= 1")
    label_smoothing = float(stages["dpo"].get("label_smoothing", 0.0))
    if not 0 <= label_smoothing < 0.5:
        raise ConfigError("dpo label_smoothing must be in [0, 0.5)")
    if int(config["project"].get("seed", -1)) < 0:
        raise ConfigError("project.seed must be >= 0")


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    project_root = config["project"].get("root", ".")
    root = Path(os.path.expandvars(project_root)).expanduser()
    if not root.is_absolute():
        root = Path(config["_config_dir"]) / root
    return (root / path).resolve()


def config_digest(config: dict[str, Any]) -> str:
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def set_global_seed(seed: int, *, include_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if include_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]
