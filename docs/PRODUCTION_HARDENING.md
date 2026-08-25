# LLM-Guided Speech Enhancement 4.0 工程夯实说明

> 状态：代码与本地 CPU 门禁阶段。本文不会把未在 AutoDL 上执行的代码写成 GPU 结果。

## 1. 目标与不变边界

项目仍然是“语言模型产生可审计的增强处方”，不是让 Qwen 直接生成波形。4.0 补齐两条
工程链路：真实波形进入策略模型，以及处方真正作用于波形并接受客观验证。

```text
真实 noisy WAV
  ├─ 冻结 Whisper encoder ─ 固定 16-token projector ─ Qwen2.5-1.5B LoRA
  │                                            └─ SFT → cDPO → GRPO
  └─ 可观测轻量特征（不含退化真值）───────────────┘
                                                 ↓
                                    有界 JSON DSP 处方
                                                 ↓
        IIR/notch/EQ/spectral suppression/dereverb/limiter executor
                                                 ↓
                   candidate WAV ─ SI-SDR/STOI/PESQ ─ accept/rollback
```

## 2. P0：波形输出腿

- `lse_v2/dsp.py` 是安全边界。未知动作、非有限参数、越过 Nyquist、破坏语音带宽的
  high-pass/low-pass、过大增益均在处理波形前失败。
- bandstop/notch 使用 SciPy IIR，EQ 使用 peaking biquad，dereverb 使用晚期混响谱估计与
  保守谱抑制，谱减和 spectral gate 有最小谱底，防止过度“挖空”。
- `lse_v2/end_to_end.py` 固化 `candidate.wav`、最终 `enhanced.wav` 和 `audit.json`。
  SI-SDR 未达到阈值会回滚为原音；无参考时只有显式允许才会输出 `unverified`。
- `lse_v2/metrics.py` 提供全局及 frame 级 SI-SDR/STOI/PESQ 增益。可选依赖不存在或
  指标失败时记录原因，不填伪零。
- `src/app.py` 的默认模型已从旧 7B 改为实际训练使用的 1.5B，并调用生产 DSP，不再把
  所有模型输出粗暴映射成同一个谱减强度。

## 3. P1：真实音频输入腿

### 3.1 物化与防泄漏

`lse_v2/materialization.py` 从许可明确的 clean/noise/RIR catalog 读取真实文件，执行真实
混噪、卷积 RIR、带宽限制并落盘 noisy WAV。split 由 speaker/source ID 哈希决定，同一
说话人不会跨 train/eval/test。每条记录保留数据集、许可、噪声和随机种子来源。

训练 prompt 只包含真实波形测得的 RMS、ZCR、谱平坦度、谱质心和削波率，不包含用于
造数的 `noise_type`、精确 SNR、RT60 和 bandlimit 真值。真值只进入 target/reward，防止
模型绕过音频直接抄 metadata。

### 3.2 原生音频条件

`lse_v2/native_training_pipeline.py` 完整执行：

1. 冻结 Whisper-small encoder；每个 WAV 的帧级 hidden state 缓存为无 pickle 的 NPZ。
2. 带 frame mask 的 adaptive pooling 把变长音频变成固定 16 个 prefix token。
3. 两层 MLP 将 Whisper 宽度映射到 Qwen hidden size。
4. prefix 与 prompt/response token embedding 拼接；prefix 和 prompt label 均为 `-100`，
   只在答案 token 上计算 causal LM loss。
5. projector 与 LoRA 一起更新，Whisper 与 Qwen 基座冻结。10% audio dropout 使用可训练
   missing-audio prefix，让模型保留可测文本特征的降级通路。

### 3.3 SFT → cDPO → GRPO

- SFT：学习合法 JSON、诊断和安全动作的基本映射。
- cDPO：同一个基座中保留冻结 reference adapter 和可训练 policy adapter，计算 chosen /
  rejected 响应的 masked sequence log-prob；使用 beta 与 label smoothing 的 conservative
  DPO，减少程序负例导致的过拟合。`native_data` 可用 `--preference-manifest` 覆盖成真实
  人工 chosen/rejected，并记录 annotator 和理由。
- GRPO：每个真实音频 prompt 在线采样多个回答，按格式、诊断、参数边界、一致性和
  过处理五项可复算 reward 做组内标准化；饱和组显式计数，零方差组不给假梯度。目标含
  clipped policy ratio 与对 DPO reference 的非负 KL 近似。

每阶段输出 LoRA、projector、artifact manifest、stage manifest 和 Accelerator/DeepSpeed
checkpoint。下一阶段从上一阶段复制 LoRA 与 projector，同时保留冻结 reference。

## 4. P2：泛化与生产

- `lse_v2/generalization.py` 在真实输出 WAV 上按 dataset/noise/SNR 切片，汇总 accept、
  rollback、指标可用样本数和平均增益。适合 VoiceBank-DEMAND、DNS、LibriTTS+DEMAND、
  WHAM! 和 Clarity，但数据下载需先人工确认许可。
- `lse_v2/streaming.py` 对 gain/IIR/notch/EQ/limiter 做带交叠淡化的窗口流式处理；需要
  全局噪声或混响状态的 spectral subtraction/dereverb 明确拒绝，避免假“实时”。
- `lse_v2/service.py` 提供带可选 Bearer token 的 FastAPI batch/stream API。公网绑定时
  没有 `LSE_API_TOKEN` 会拒绝启动。支持 vLLM/SGLang 的 OpenAI-compatible 文本兜底。
- `lse_v2/native_inference.py` 提供真实 Whisper-prefix + LoRA 本地 CUDA 推理；自定义连续
  prefix 目前不能原样塞入标准 vLLM，因此 vLLM 路径被明确标为文本 fallback。
- `lse_v2/listening_test.py` 从真实审计 WAV 生成固定随机种子的双盲 A/B 包，将盲底与
  试听清单分开保存，并汇总候选胜率、平局率和自然度/可懂度/噪声残留/音乐噪声评分。

## 5. DeepSpeed 与单卡边界

默认配置是 QLoRA + ZeRO-2 合同。单张 4090 上 ZeRO 不会产生跨 GPU 分片收益；它的作用
是验证团队常见 DeepSpeed 启动、状态保存和恢复接口。ZeRO-3 在 world size 1 同样不是
“单卡并行”，并可能比 ZeRO-2 更慢。只有实际扩展到多卡后，才能声称分布式加速或分片。

## 6. 验收口径

本地完成条件：全部单测、Ruff、格式、配置 dry-run 与 wheel 构建通过。GPU 完成条件：

- 物化音频记录数、source split 和许可报告存在；
- Whisper cache 的 encoded/reused 数量可核；
- SFT、cDPO、GRPO 均产生非有限检查通过的 loss 和独立 checkpoint；
- GRPO 记录 reward 与 saturated group 数，不以单一 reward 代替波形质量；
- test/OOD 集输出真实 candidate/enhanced WAV、全局与 frame 指标、rollback 率；
- 记录峰值 VRAM、吞吐和 P50/P95 延迟；
- 人工听测按盲测表记录，不用主观挑例代替统计结果。
