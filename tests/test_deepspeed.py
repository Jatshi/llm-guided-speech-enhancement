from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lse_v2.config import load_config
from lse_v2.contracts import build_alignment_datasets
from lse_v2.deepspeed import (
    DeepSpeedContractError,
    resolve_deepspeed,
    validate_deepspeed_config,
)
from lse_v2.training import (
    DEEPSPEED_DISTRIBUTED_ENV,
    _ensure_single_process_deepspeed_env,
    build_parser,
    train_stage,
)

REPO = Path(__file__).resolve().parents[1]
ZERO2 = REPO / "configs" / "deepspeed" / "ds_zero2.json"
ZERO3 = REPO / "configs" / "deepspeed" / "ds_zero3_offload.json"


def load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "stage", "offload"),
    [(ZERO2, 2, False), (ZERO3, 3, True)],
)
def test_checked_in_profiles_are_single_gpu_contracts(
    path: Path, stage: int, offload: bool
) -> None:
    result = validate_deepspeed_config(load_profile(path), actual_world_size=1)
    assert result["stage"] == stage
    assert result["declared_world_size"] == 1
    assert result["offload_param"] is offload
    assert "_portfolio_contract" not in result["runtime_config"]


def test_world_size_mismatch_is_rejected() -> None:
    with pytest.raises(DeepSpeedContractError, match="WORLD_SIZE=2"):
        validate_deepspeed_config(load_profile(ZERO2), actual_world_size=2)


def test_invalid_precision_and_offload_combinations_are_rejected() -> None:
    payload = load_profile(ZERO2)
    payload["bf16"]["enabled"] = True
    payload["fp16"]["enabled"] = True
    with pytest.raises(DeepSpeedContractError, match="cannot both"):
        validate_deepspeed_config(payload, actual_world_size=1)

    payload = load_profile(ZERO2)
    payload["zero_optimization"]["offload_param"] = {"device": "cpu"}
    with pytest.raises(DeepSpeedContractError, match="stage 3"):
        validate_deepspeed_config(payload, actual_world_size=1)


def test_cli_none_disables_configured_deepspeed() -> None:
    config = load_config(REPO / "configs" / "autodl_4090.json")
    assert resolve_deepspeed(config, "sft", actual_world_size=1)["stage"] == 2
    assert resolve_deepspeed(config, "sft", cli_value="none", actual_world_size=1) is None


@pytest.mark.parametrize("flag", ["--local_rank=0", "--local-rank=0"])
def test_training_cli_accepts_deepspeed_local_rank(flag: str) -> None:
    args = build_parser().parse_args([flag, "--stage", "sft", "--dry-run"])
    assert args.local_rank == 0
    assert args.stage == "sft"


def test_single_process_deepspeed_env_avoids_mpi_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in DEEPSPEED_DISTRIBUTED_ENV:
        monkeypatch.delenv(key, raising=False)
    environment = _ensure_single_process_deepspeed_env(1)
    assert environment == {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }


def test_single_process_deepspeed_env_preserves_launcher_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MASTER_PORT", "29600")
    monkeypatch.setenv("LOCAL_RANK", "7")
    environment = _ensure_single_process_deepspeed_env(1)
    assert environment["MASTER_PORT"] == "29600"
    assert environment["LOCAL_RANK"] == "7"


def test_training_dry_run_records_deepspeed_mapping(tmp_path: Path) -> None:
    build_alignment_datasets(REPO / "examples" / "audio_manifest.smoke.jsonl", tmp_path / "data")
    config = json.loads((REPO / "configs" / "smoke.json").read_text(encoding="utf-8"))
    config["project"]["root"] = str(tmp_path)
    for stage in ("sft", "dpo", "grpo"):
        config["data"][f"{stage}_train"] = f"data/{stage}/train.jsonl"
        config["data"][f"{stage}_eval"] = f"data/{stage}/eval.jsonl"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = train_stage(
        config_path,
        "sft",
        dry_run=True,
        deepspeed_override=str(ZERO3),
    )

    assert result["distributed"]["world_size"] == 1
    assert result["distributed"]["deepspeed"]["stage"] == 3
    assert result["distributed"]["deepspeed"]["offload_param"] is True


def test_cpu_multiprocess_contract_smoke_makes_no_gpu_claim() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "distributed_contract_smoke.py"),
            "--world-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert report["ranks"] == [0, 1]
    assert report["multi_gpu_claim"] is False
    assert report["transport"] == "local CPU multiprocessing only"
