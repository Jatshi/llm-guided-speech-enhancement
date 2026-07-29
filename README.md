# LLM-Guided Speech Enhancement 2.0

[![CI](https://github.com/Jatshi/llm-guided-speech-enhancement/actions/workflows/ci.yml/badge.svg)](https://github.com/Jatshi/llm-guided-speech-enhancement/actions/workflows/ci.yml)

[从零手搓完整学习手册](docs/audio_llm_v2_from_scratch_zh.md) ·
[项目实施计划](PROJECT_PLAN.md) ·
[架构说明](docs/architecture.md) ·
[最终GRPO LoRA（Hugging Face）](https://huggingface.co/jatshi/Audio-Codec-LLM-Qwen2.5-1.5B-GRPO-LoRA)

一个面向单张RTX 4090的、可复现的语音增强策略大模型训练项目。2.0将原有硬编码
SFT/DPO脚本升级为完整的：

```text
音频/声学证据清单
  → 版本化SFT数据
  → 版本化DPO偏好对
  → 可验证GRPO奖励
  → 离线预测、奖励消融与机器可读报告
```

本项目输出的是可审计的增强处方，而不是把语言模型伪装成端到端波形修复器。每条处方
必须包含退化诊断、可执行DSP动作、理由和置信度。音频路径、直接提取的声学特征和标签
均保存在版本化manifest中。

## 2.0新增能力

- 统一JSON/YAML配置，支持CLI覆盖、固定随机种子和自动断点续训。
- 保留`src/train_sft.py`、`src/train_dpo.py`等旧命令，但实现已迁移到`lse_v2`。
- `lse.audio_manifest.v2`记录真实音频路径、采样率、时长、声学特征、目标处方和来源。
- 自动生成`lse.sft.v2`、`lse.dpo.v2`、`lse.grpo.v2`三个数据契约。
- GRPO采用可离线复算的五项奖励：格式、诊断、参数范围、一致性、过处理惩罚。
- SFT → DPO → GRPO完整训练管线，默认Qwen2.5-1.5B + LoRA，适配24GB显存。
- 训练结束后生成真实模型预测，再输出奖励分量、违规类型与奖励消融JSON报告。
- SFT/DPO/GRPO均支持DeepSpeed ZeRO-2和ZeRO-3 CPU offload配置与CLI覆盖。
- AutoDL统一提供preflight、bootstrap、run三个脚本，支持`smoke`和`full`模式。

## 本地CPU烟雾验证

烟雾验证不会下载模型、不会使用GPU、不会声称产生模型指标：

```bash
python -m pip install -e ".[test]"
PORTFOLIO_V2_MODE=smoke bash scripts/autodl_v2_run.sh
```

它会验证：

1. 示例音频manifest；
2. 三阶段数据构建；
3. SFT/DPO/GRPO配置和数据契约；
4. 断点续训计划与产物路径；
5. 参考处方奖励及五项消融；
6. 全部离线单元测试。

Windows PowerShell可使用等价命令：

```powershell
python -m lse_v2.data_cli build `
  --manifest examples/audio_manifest.smoke.jsonl `
  --output-dir outputs/smoke/data
python -m lse_v2.pipeline --config configs/smoke.json --dry-run
python -m pytest
```

## AutoDL 4090直接运行

仓库上传到AutoDL后，只需：

```bash
cd /root/autodl-tmp/audio-codec-llm
export PORTFOLIO_V2_MODE=full
bash scripts/autodl_v2_bootstrap.sh
bash scripts/autodl_v2_run.sh
```

如果已有v2清单：

```bash
export AUDIO_MANIFEST=/root/autodl-tmp/datasets/audio_manifest.v2.jsonl
bash scripts/autodl_v2_run.sh
```

如果仓库中存在旧版`data/training/metadata.json`，run脚本会自动迁移。否则它会在训练
前明确退出，不会空烧GPU。

### AutoDL默认资源策略

- 基座：`Qwen/Qwen2.5-1.5B-Instruct`
- 精度：BF16
- 微调：LoRA，`r=16`
- SFT/DPO：每卡batch 8，梯度累积2，有效batch 16
- DPO：`beta=0.01`、`label_smoothing=0.1` 的 conservative DPO；参数来自全量初跑的偏好margin饱和诊断
- GRPO：每卡batch 4，梯度累积4，2个候选，最大生成256 tokens
- DeepSpeed：默认ZeRO-2；可切换ZeRO-3 CPU offload
- 检查点：每阶段最多保留2个，`--resume auto`自动恢复
- 预检：至少20GB显存、35GB磁盘、CUDA可见

配置位于[configs/autodl_4090.json](configs/autodl_4090.json)。3B模型可以通过修改
`model.name_or_path`启用，但必须重新执行preflight并观察GRPO峰值显存。

## 数据准备

### 从旧metadata迁移

```bash
python -m lse_v2.data_cli migrate \
  --legacy-metadata data/training/metadata.json \
  --output data/training/audio_manifest.v2.jsonl
```

如已安装`.[audio]`，可加入`--extract-features`直接读取音频并提取RMS、频谱平坦度、
频谱质心、过零率和时长。

旧记录只有`clean_path`而没有物化的`audio_path`时，迁移器会明确标记
`source_role=clean_proxy_for_synthetic_degradation`，不会把它伪装成真实噪声音频；
这类样本的声学标签来自退化配置，直接音频特征留空。

### 验证真实音频存在

```bash
python -m lse_v2.data_cli validate \
  --manifest data/training/audio_manifest.v2.jsonl \
  --check-audio-files
```

### 构建三阶段数据

```bash
python -m lse_v2.data_cli build \
  --manifest data/training/audio_manifest.v2.jsonl \
  --output-dir data/v2 \
  --seed 42 \
  --eval-ratio 0.05
```

## 分阶段运行

```bash
python -m lse_v2.training --config configs/autodl_4090.json --stage sft --resume auto
python -m lse_v2.training --config configs/autodl_4090.json --stage dpo --resume auto
python -m lse_v2.training --config configs/autodl_4090.json --stage grpo --resume auto
```

显式选择或关闭DeepSpeed：

```bash
# 单卡默认ZeRO-2
python -m lse_v2.training \
  --config configs/autodl_4090.json \
  --stage sft \
  --deepspeed configs/deepspeed/ds_zero2.json

# 显存不足时用CPU内存换吞吐
python -m lse_v2.training \
  --config configs/autodl_4090.json \
  --stage sft \
  --deepspeed configs/deepspeed/ds_zero3_offload.json

# 运行无DeepSpeed对照
python -m lse_v2.training \
  --config configs/autodl_4090.json \
  --stage sft \
  --deepspeed none
```

只验证、不加载模型：

```bash
python -m lse_v2.training \
  --config configs/autodl_4090.json \
  --stage grpo \
  --dry-run
```

### DeepSpeed证据边界

仓库内两个DeepSpeed配置都声明`world_size=1`并标记为`single_gpu_only`：

- [ds_zero2.json](configs/deepspeed/ds_zero2.json)：验证DeepSpeed运行兼容性，并作为性能
  对照；**单卡不会获得跨GPU参数或优化器状态分片收益**。
- [ds_zero3_offload.json](configs/deepspeed/ds_zero3_offload.json)：把参数和优化器状态
  offload到CPU，可能降低峰值显存，但通常牺牲吞吐；结论必须实测。

单GPU真实一步smoke：

```bash
export PORTFOLIO_V2_MODE=full
bash scripts/autodl_v2_bootstrap.sh
bash scripts/deepspeed_single_gpu_smoke.sh
```

CPU多进程契约smoke：

```bash
python scripts/distributed_contract_smoke.py --world-size 2
```

后者只验证rank、local rank和world size传播，使用本机CPU multiprocessing；它没有执行
NCCL、DeepSpeed训练或多GPU通信，不能作为多GPU运行证据。

显存与吞吐实验必须填写
[deepspeed_comparison_template.csv](docs/deepspeed_comparison_template.csv)，不得预填或凭
理论推断。

## 奖励与离线评测

GRPO总奖励是五个可验证分量的加权平均：

| 分量 | 验证内容 |
| --- | --- |
| `format` | 是否为合法JSON，是否包含四个必需字段 |
| `diagnosis` | 噪声、混响等诊断是否与声学证据一致 |
| `parameter_bounds` | dB、Q、Hz等DSP参数是否处于安全范围 |
| `consistency` | 诊断与动作是否相互支持，是否伤害语音频带 |
| `overprocessing` | 是否存在过强衰减、累计抑制或高SNR过处理 |

训练完成后，管线会用最终GRPO adapter在独立eval集生成
`outputs/v2/evaluation/predictions.jsonl`，然后生成：

```text
outputs/v2/evaluation/reward_report.json
```

如果没有提供模型预测，评测器只会执行“参考答案奖励自检”，并在报告中明确注明
“not a model benchmark”，不会伪造模型结果。

独立评测命令：

```bash
python -m lse_v2.evaluation \
  --dataset data/v2/grpo/eval.jsonl \
  --predictions outputs/v2/evaluation/predictions.jsonl \
  --output outputs/v2/evaluation/reward_report.json
```

## 产物

每阶段均输出：

```text
outputs/v2/<stage>/
├── checkpoint-*/
├── final/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── tensorboard/
└── stage_manifest.json
```

`stage_manifest.json`记录配置哈希、代码commit、依赖版本、随机种子、恢复点、输入数量、
状态和产物路径。总控状态保存在`outputs/v2/pipeline_status.json`。

## 测试

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall -q lse_v2 src
python scripts/distributed_contract_smoke.py --world-size 2
```

测试完全离线，不需要模型权重或GPU。

## 真实性边界

- 仓库内的smoke报告只证明数据、奖励、配置和管线能运行，不是GPU训练结果。
- README不预填PESQ、STOI、DNSMOS或模型胜率；这些数字必须来自之后的真实实验。
- 当前语言模型根据音频manifest中的声学证据生成策略，并非直接编码原始波形的
  AudioLM。要升级到波形token或音频编码器，需要另立模型结构与基线，不能只改项目名。
- 程序生成的偏好负例用于启动训练，公开发布前应加入人工复核的真实偏好对。

完整实施与验收计划见[PROJECT_PLAN.md](PROJECT_PLAN.md)，可复制命令和待填GPU结果见
[run_manifest.md](run_manifest.md)。

最终 GRPO 评测之外，发布前必须在同一留出集上运行四阶段矩阵：

```bash
python scripts/evaluate_stage_matrix.py \
  --config configs/autodl_4090.json \
  --max-samples 200 \
  --batch-size 8
```

它分别保存 base、SFT、DPO、GRPO 的真实生成、五分量 reward、
推理吞吐和 peak VRAM。`evaluate_rewards` 在没有 prediction 时只是在校验
reference/reward 合同，不能代替该模型矩阵。

发布前先运行强制验收；只有GRPO完成、四阶段矩阵完整且模型卡无待填标记时才能上传：

```bash
python scripts/publish_hf_adapter.py --dry-run
python scripts/publish_hf_adapter.py
```

真实上传要求通过环境变量设置`HF_TOKEN`，脚本不接受命令行token，避免密钥进入shell
history。

## License

MIT。使用Qwen、AISHELL或其他语音数据时，还需分别遵守其模型与数据许可证。
