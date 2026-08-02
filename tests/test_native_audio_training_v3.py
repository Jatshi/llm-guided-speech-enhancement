from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lse_v2.native_audio_training import load_native_audio_manifest


def test_native_audio_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    audio = tmp_path / "sample.wav"
    soundfile.write(audio, np.zeros(160, dtype=np.float32), 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"sample_id": "one", "audio_path": audio.name, "target_text": "{\"status\":\"ok\"}"}
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_native_audio_manifest(manifest)
    assert records[0].audio_path == audio.resolve()


def test_native_audio_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    row = {"sample_id": "dup", "audio_path": audio.name, "target_text": "target"}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_native_audio_manifest(manifest)
