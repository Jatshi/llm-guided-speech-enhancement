from __future__ import annotations

import json
from pathlib import Path

from lse_v2.catalog import build_audio_catalog


def test_audio_catalog_has_license_and_source_identity(tmp_path: Path) -> None:
    speaker = tmp_path / "speaker-1"
    speaker.mkdir()
    (speaker / "one.wav").write_bytes(b"placeholder")
    output = tmp_path / "catalog.jsonl"

    report = build_audio_catalog(
        tmp_path,
        output,
        dataset="local-clean",
        license_name="CC-BY-4.0",
        role="clean",
    )
    row = json.loads(output.read_text(encoding="utf-8"))

    assert report["records"] == 1
    assert row["source_id"] == "speaker-1/one"
    assert row["speaker_id"] == "speaker-1"
    assert row["license"] == "CC-BY-4.0"
