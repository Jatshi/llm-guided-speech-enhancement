"""Idempotently download the two public base models required by native v4.x."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {
    "whisper-small": "openai/whisper-small",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}


def model_is_complete(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    weight_patterns = ("*.safetensors", "*.bin")
    return any(any(path.glob(pattern)) for pattern in weight_patterns)


def ensure_models(root: Path, *, endpoint: str | None, max_workers: int) -> dict:
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    reused: list[str] = []
    for directory, repo_id in MODELS.items():
        target = root / directory
        if model_is_complete(target):
            reused.append(repo_id)
            continue
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            endpoint=endpoint,
            max_workers=max_workers,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
        if not model_is_complete(target):
            raise RuntimeError(f"download completed without usable weights: {target}")
        downloaded.append(repo_id)
    marker = root / ".models_download_complete"
    marker.write_text("\n".join(MODELS.values()) + "\n", encoding="utf-8")
    return {
        "status": "ready",
        "root": str(root.resolve()),
        "downloaded": downloaded,
        "reused": reused,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/lse-v4-models"))
    parser.add_argument("--endpoint")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    print(
        json.dumps(
            ensure_models(args.root, endpoint=args.endpoint, max_workers=args.max_workers),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
