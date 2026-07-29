from __future__ import annotations

import importlib.util
from pathlib import Path


def test_legacy_entry_points_compile_and_expose_main() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    for name in ("train_sft.py", "train_dpo.py", "train_grpo.py", "build_llm_data.py"):
        spec = importlib.util.spec_from_file_location(name, root / name)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
