# -*- coding: utf-8 -*-
"""从 ModelScope 下载 Qwen2.5-7B-Instruct 到系统盘 /root/models（国内高速）。"""
import os
import sys

TARGET = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/Qwen2.5-7B-Instruct"))


def main():
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    from modelscope import snapshot_download
    # ModelScope 上 Qwen 官方仓库
    path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir=TARGET)
    print("MODEL_DOWNLOADED_TO:", path)
    # 简单校验关键文件
    need = ["config.json", "tokenizer.json"]
    for f in need:
        p = os.path.join(TARGET, f)
        print(f, "OK" if os.path.exists(p) else "MISSING")
    # 列出 safetensors 分片
    shards = [x for x in os.listdir(TARGET) if x.endswith(".safetensors")]
    print("SAFETENSORS_SHARDS:", len(shards))


if __name__ == "__main__":
    main()
