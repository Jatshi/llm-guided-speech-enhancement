"""
生成训练用退化配置元数据（磁盘友好）。

- 从已有 AISHELL-1 采样 N 个干净语音文件；
- 每个文件生成 3 个随机退化配置（仅配置，不落盘音频）；
- 仅对前 EVAL_SUBSET 个样本真正生成并保存退化 wav，用于评估/Demo；
- 输出 metadata.json 供 build_llm_data.py 使用。
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degradation import Degradation, sample_config

_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AISHELL_WAV = os.environ.get("AISHELL_WAV", os.path.join(_ROOT, "data/raw/aishell/wav"))
OUT_DIR = os.path.join(_ROOT, "data/training")
EVAL_AUDIO_DIR = os.path.join(_ROOT, "data/processed/eval_audio")


def list_wavs(root, limit):
    """列出 AISHELL wav 文件（train 优先），限制数量"""
    root = Path(root)
    train_dir = root / "train"
    scan = train_dir if train_dir.exists() else root
    files = []
    for p in sorted(scan.rglob("*.wav")):
        files.append(str(p))
        if len(files) >= limit * 3:  # 多收集一些以便打乱
            break
    random.shuffle(files)
    return files[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_files", type=int, default=40000, help="采样的干净语音文件数")
    ap.add_argument("--deg_per_file", type=int, default=3, help="每个文件生成的退化配置数")
    ap.add_argument(
        "--eval_subset", type=int, default=200, help="真正落盘退化音频的样本数(评估/Demo)"
    )
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    rng = random.Random(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EVAL_AUDIO_DIR, exist_ok=True)

    print(f"扫描 AISHELL 干净语音: {AISHELL_WAV}")
    wavs = list_wavs(AISHELL_WAV, args.num_files)
    print(f"采样得到 {len(wavs)} 个干净语音文件，每个生成 {args.deg_per_file} 个退化配置")
    print(f"预计训练样本总数: {len(wavs) * args.deg_per_file}")

    degrader = Degradation(sr=16000)
    samples = []
    saved_eval = 0

    for idx, wav in enumerate(tqdm(wavs, desc="生成配置")):
        for d in range(args.deg_per_file):
            cfg = sample_config(rng)
            rec = {
                "id": f"{idx:06d}_{d}",
                "clean_path": wav,
                "degradation_config": cfg,
                "audio_path": None,
                "provenance": {
                    "dataset": "AISHELL-1",
                    "catalog": "OpenSLR SLR33",
                    "source_url": "https://www.openslr.org/resources/33/data_aishell.tgz",
                    "source_md5": "2f494334227864a8a8fec932999db9d8",
                    "license": "Apache-2.0",
                    "sampling_seed": 42,
                },
            }
            # 仅前 eval_subset 个样本真正生成退化音频落盘
            if saved_eval < args.eval_subset and d == 0:
                try:
                    audio, _ = librosa.load(wav, sr=16000, duration=5.0)
                    if len(audio) >= 8000:
                        degraded = degrader.apply(audio, cfg)
                        out_wav = os.path.join(EVAL_AUDIO_DIR, f"eval_{saved_eval:04d}.wav")
                        sf.write(out_wav, degraded, 16000)
                        rec["audio_path"] = out_wav
                        saved_eval += 1
                except Exception:
                    pass
            samples.append(rec)

    meta_path = os.path.join(OUT_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    metadata_sha256 = hashlib.sha256(Path(meta_path).read_bytes()).hexdigest()
    provenance_path = os.path.join(OUT_DIR, "dataset_provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "lse.dataset_provenance.v2",
                "dataset": "AISHELL-1",
                "catalog": "OpenSLR SLR33",
                "source_url": "https://www.openslr.org/resources/33/data_aishell.tgz",
                "source_md5": "2f494334227864a8a8fec932999db9d8",
                "license": "Apache-2.0",
                "aishell_wav_root": str(Path(AISHELL_WAV).resolve()),
                "sampling_seed": 42,
                "requested_clean_files": args.num_files,
                "sampled_clean_files": len(wavs),
                "degradations_per_file": args.deg_per_file,
                "records": len(samples),
                "materialized_noisy_audio_records": saved_eval,
                "metadata_sha256": metadata_sha256,
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(f"完成！总样本 {len(samples)}，落盘评估音频 {saved_eval} 条")
    print(f"元数据: {meta_path}")


if __name__ == "__main__":
    main()
