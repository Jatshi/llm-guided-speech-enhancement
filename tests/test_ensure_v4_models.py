from __future__ import annotations

from pathlib import Path


def test_model_is_complete_requires_config_and_weights(tmp_path: Path) -> None:
    from scripts.ensure_v4_models import model_is_complete

    assert not model_is_complete(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not model_is_complete(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert model_is_complete(tmp_path)
