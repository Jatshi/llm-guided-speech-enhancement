"""DeepSpeed configuration resolution and contract validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path


class DeepSpeedContractError(ValueError):
    """Raised before launch when a DeepSpeed configuration is contradictory."""


def validate_deepspeed_config(payload: dict[str, Any], *, actual_world_size: int) -> dict[str, Any]:
    if actual_world_size < 1:
        raise DeepSpeedContractError("actual_world_size must be >= 1")
    zero = payload.get("zero_optimization")
    if not isinstance(zero, dict):
        raise DeepSpeedContractError("zero_optimization must be an object")
    stage = zero.get("stage")
    if stage not in {0, 1, 2, 3}:
        raise DeepSpeedContractError("zero_optimization.stage must be 0, 1, 2, or 3")
    contract = payload.get("_portfolio_contract")
    if not isinstance(contract, dict):
        raise DeepSpeedContractError("Missing _portfolio_contract metadata")
    declared = contract.get("declared_world_size")
    if not isinstance(declared, int) or declared < 1:
        raise DeepSpeedContractError(
            "_portfolio_contract.declared_world_size must be a positive integer"
        )
    if declared != actual_world_size:
        raise DeepSpeedContractError(
            f"DeepSpeed world-size contract declares {declared}, "
            f"but launcher reports WORLD_SIZE={actual_world_size}"
        )
    bf16_enabled = payload.get("bf16", {}).get("enabled")
    fp16_enabled = payload.get("fp16", {}).get("enabled")
    if bf16_enabled is True and fp16_enabled is True:
        raise DeepSpeedContractError("bf16 and fp16 cannot both be enabled")
    offload_optimizer = zero.get("offload_optimizer")
    offload_param = zero.get("offload_param")
    if stage != 3 and offload_param:
        raise DeepSpeedContractError("Parameter offload is only valid with ZeRO stage 3")
    for name, item in (
        ("offload_optimizer", offload_optimizer),
        ("offload_param", offload_param),
    ):
        if item is not None and (
            not isinstance(item, dict) or item.get("device") not in {"cpu", "nvme"}
        ):
            raise DeepSpeedContractError(f"{name}.device must be 'cpu' or 'nvme'")
    if contract.get("single_gpu_only") is True and actual_world_size != 1:
        raise DeepSpeedContractError("This checked-in config is explicitly single-GPU only")
    runtime = {key: value for key, value in payload.items() if key != "_portfolio_contract"}
    return {
        "runtime_config": runtime,
        "stage": stage,
        "declared_world_size": declared,
        "offload_optimizer": bool(offload_optimizer),
        "offload_param": bool(offload_param),
        "single_gpu_note": contract.get("single_gpu_note"),
    }


def resolve_deepspeed(
    config: dict[str, Any],
    stage: str,
    *,
    cli_value: str | None = None,
    actual_world_size: int = 1,
) -> dict[str, Any] | None:
    configured = config["training"]["stages"][stage].get("deepspeed")
    selected = configured if cli_value is None else cli_value
    if selected in {None, "", False, "none", "off"}:
        return None
    path = resolve_path(config, str(selected))
    if not path.is_file():
        raise DeepSpeedContractError(f"DeepSpeed config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeepSpeedContractError("DeepSpeed config must be a JSON object")
    validated = validate_deepspeed_config(payload, actual_world_size=actual_world_size)
    validated["source_path"] = str(path)
    return validated


def write_runtime_config(resolved: dict[str, Any], output_dir: str | Path) -> Path:
    target = Path(output_dir) / "deepspeed_runtime.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(resolved["runtime_config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        action="append",
        choices=("sft", "dpo", "grpo"),
        required=True,
    )
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    profiles: dict[str, Any] = {}
    for stage in args.stage:
        resolved = resolve_deepspeed(config, stage, actual_world_size=args.world_size)
        profiles[stage] = (
            None
            if resolved is None
            else {key: value for key, value in resolved.items() if key != "runtime_config"}
        )
    print(json.dumps({"valid": True, "profiles": profiles}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
