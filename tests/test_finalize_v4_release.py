from __future__ import annotations

import json
from pathlib import Path

from scripts.finalize_v4_release import count_jsonl, sha256_file


def test_count_jsonl_validates_and_counts_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    assert count_jsonl(path) == 2


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"evidence-backed-v4")
    assert sha256_file(path) == "e09e3246c486666175f8442b9d8be590ad53db829ff12868cdb2c5b7dadf46e5"


def test_count_jsonl_rejects_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": 1}) + "\nnot-json\n", encoding="utf-8")
    try:
        count_jsonl(path)
    except ValueError as exc:
        assert "invalid JSONL" in str(exc)
    else:
        raise AssertionError("invalid JSONL must be rejected")
