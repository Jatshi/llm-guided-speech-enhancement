# AutoDL 4090 运行手册（4.0）

## 开机前已准备好的内容

- 生产 DSP、客观指标、回滚审计、流式处理和服务代码；
- 真实音频物化、许可 catalog、source-level split；
- 冻结 Whisper cache、audio projector、QLoRA SFT/cDPO/GRPO；
- 单卡 DeepSpeed ZeRO-2 配置、checkpoint/resume、运行清单；
- CPU 单元测试与静态检查。GPU 数字仍为空，等 AutoDL 实跑填入。

## 1. 数据目录

建议把大文件放 `/root/autodl-tmp/lse-v4-data`，仓库只保存 manifest 和小型报告。先为
clean/noise/RIR 各建一个许可明确的 catalog：

```bash
python -m lse_v2.catalog --input-dir /root/autodl-tmp/datasets/clean \
  --output /root/autodl-tmp/lse-v4-data/clean.jsonl \
  --dataset DATASET_NAME --license LICENSE_ID --role clean

python -m lse_v2.catalog --input-dir /root/autodl-tmp/datasets/noise \
  --output /root/autodl-tmp/lse-v4-data/noise.jsonl \
  --dataset DATASET_NAME --license LICENSE_ID --role noise
```

不得为了省事把未知许可写成 CC0。RIR 不存在时可以先使用程序化衰减响应，但报告会如实
标记 generated，不会冒充实测 RIR。

## 2. 一次性安装与预检

```bash
cd /root/autodl-tmp/llm-guided-speech-enhancement
bash scripts/autodl_v4_bootstrap.sh
source /root/autodl-tmp/lse-v4-env/bin/activate
bash scripts/autodl_v4_preflight.sh
```

预检要求 CUDA 可见、至少 20 GiB VRAM、数据盘至少 80 GiB 可用、依赖一致、Hugging Face
可达。任何一项失败都在模型下载和训练前退出。

## 3. 物化 1–2 万条真实 noisy WAV

```bash
export LSE_CLEAN_MANIFEST=/root/autodl-tmp/lse-v4-data/clean.jsonl
export LSE_NOISE_MANIFEST=/root/autodl-tmp/lse-v4-data/noise.jsonl
export LSE_RIR_MANIFEST=/root/autodl-tmp/lse-v4-data/rir.jsonl  # 可选
export LSE_MAX_SOURCES=10000
export LSE_VARIANTS_PER_SOURCE=2
bash scripts/autodl_v4_prepare_data.sh
```

先查看：

```bash
jq . /root/autodl-tmp/lse-v4-data/materialized/materialization_report.json
jq . data/native/native_audio.v1.report.json
```

确认 train/eval/test 都非零，`all_audio_materialized=true`，同一 source 没有跨 split。

## 4. 训练

```bash
export LSE_RUN_MODE=full
bash scripts/autodl_v4_run.sh 2>&1 | tee outputs/native_v4/console.log
```

断线后重新执行同一命令，会从各 stage 最新 `checkpoint-*` 恢复。不要删除前一阶段
`final/adapter` 和 `audio_projector.pt`，它们是下一阶段 reference 的来源。

如果 4-bit 与当前镜像 DeepSpeed 组合报兼容错误，先记录完整错误和版本，再把对应 stage
的 `deepspeed` 字段删除，仅用 Accelerate+QLoRA 继续；不要同时盲改 torch、CUDA、
transformers 和 bitsandbytes。单卡下这不影响“不是多卡训练”的事实边界。

## 5. 运行后必须检查

```bash
jq . outputs/native_v4/run_manifest.json
find outputs/native_v4 -maxdepth 4 -name 'stage_manifest.json' -o -name 'artifact_manifest.json'
nvidia-smi
```

然后在保留的 test/OOD split 上生成 prescription，执行：

```bash
python -m lse_v2.generalization \
  --manifest /path/to/test_audio_manifest.jsonl \
  --predictions /path/to/predictions.jsonl \
  --output-dir outputs/native_v4/generalization
```

最终只有 `run_manifest.json`、三阶段产物、真实 WAV 审计和 OOD 报告全在，才算完整链路，
仅跑若干 step 的 smoke 只能叫链路测试。

## 6. 服务

本机回环：

```bash
python -m lse_v2.service --host 127.0.0.1 --port 8000
```

公网或 AutoDL 代理端口：

```bash
export LSE_API_TOKEN='生成一个足够长的随机 token'
python -m lse_v2.service --host 0.0.0.0 --port 8000
```

标准 vLLM/SGLang endpoint 只能走文本特征 fallback。要用真实音频 prefix，加载
`NativeAudioPlanner` 本地 CUDA 后注入 `create_app(planner=...)`。
