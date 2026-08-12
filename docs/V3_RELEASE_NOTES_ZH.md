# LLM-Guided Speech Enhancement 3.0 新增内容与发布说明

## 版本定位

2.0 让文本 LLM 根据结构化声学证据生成增强处方；3.0 补上两块关键缺口：直接从原始
音频构造连续条件，以及执行增强后重新测量并决定接受、修订或回滚。项目从“一次性
参数预测”变成可验证闭环，但仍不把语言模型伪装成端到端波形生成器。

## 新增能力

### 1. 原生音频条件

- 新增 `WhisperAudioProjector`；
- 冻结 Whisper encoder，将变长 hidden states 做 masked pooling；
- 两层 MLP 映射到 Qwen embedding 维度并形成 audio prefix；
- padding mask 同步扩展，避免无效帧污染条件表示；
- 训练初期只更新约 7.1MB projector，降低显存与过拟合风险。

### 2. 闭环增强控制器

- `NumpyDSPExecutor` 实现有界增益、去直流和 spectral gate；
- 动作参数越界时 fail closed；
- 完整流程为 execute → measure → revise → re-measure → accept/rollback；
- metric snapshot 支持 SI-SDR、STOI、PESQ、DNSMOS、WER 和 speaker similarity；
- 缺失指标返回 `unavailable + reason`，不以 0 冒充测量值。

### 3. 训练与分布式合同

- 新增原生音频数据 manifest 与训练入口；
- 增加 DeepSpeed ZeRO-2、ZeRO-3/offload 配置；
- 单卡验证 rank、seed、checkpoint 和恢复路径；
- 明确 world size=1 不是多卡并行加速证据。

### 4. 权重与文档发布

- 导出 `audio_projector.pt` 和机器可读 run manifest；
- 上传 [Native Audio Projector v3](https://huggingface.co/jatshi/Audio-Codec-LLM-Native-Audio-Projector-v3)；
- README 保留既有 GRPO LoRA，并增加 v3 projector 直达链接；
- 包版本、README 和运行时版本统一为 `3.0.0`。

## RTX 4090 验证

| 项目 | 结果 |
|---|---:|
| 模型 | Whisper-small + Qwen2.5-1.5B-Instruct |
| optimizer steps | 2 |
| loss | 3.31035 → 2.70892 |
| mean loss | 3.00964 |
| 峰值显存 | 3,859.58 MiB |
| projector | 7,089,880 字节 |
| 本地回归 | 51 tests passed |

两步 loss 下降证明音频编码、prefix 注入、反向传播、优化器与保存链路工作；它不证明
模型收敛，也不证明 PESQ、WER 或主观音质提升。

## 主要新增文件

```text
lse_v2/audio_conditioning.py       音频池化与 projector
lse_v2/native_audio_training.py    原生音频训练循环
lse_v2/closed_loop.py              执行—测量—回滚闭环
scripts/train_native_audio_v3.py   训练入口
scripts/build_native_audio_smoke_v3.py
configs/v3_closed_loop_4090.json
```

## 声明边界

`WhisperAudioProjector` 是连续条件映射器，不具备离散码本、码率或音频重建目标，因此
不能称为真正的 neural audio codec。项目沿用历史名称，但文档和面试必须说清模块语义。

详细数学、闭环设计、DeepSpeed 解释和工程排障见
[3.0 学习与踩坑手册](V3_LEARNING_AND_INTERVIEW_ZH.md) 与
[完整开发复盘](V3_DEVELOPMENT.md)。
