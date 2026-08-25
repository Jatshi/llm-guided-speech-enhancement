# LLM-Guided Speech Enhancement 4.0 发布说明

## 版本定位

4.0 的目标不是再做一层漂亮 Demo，而是把“LLM 生成语音增强处方”夯实为可训练、可执行、
可测量、可回滚、可恢复和可审计的完整工程。系统接收真实噪声音频和辅助声学证据，输出
结构化 DSP 处方；处方经过严格校验后作用于波形，并以客观指标决定接受还是回滚。

```text
物化噪声音频
  → frozen Whisper encoder
  → 16 个连续音频前缀 token
  → Qwen2.5-1.5B + LoRA
  → JSON 增强处方
  → 参数白名单与边界校验
  → 可执行 DSP
  → SI-SDR / PESQ / STOI 复测
  → 接受或回滚
```

## 数据

- 干净语音：LibriSpeech `train-clean-100`。
- 环境噪声：ESC-50。
- 物化记录：20,000 条真实 noisy WAV，而不是只把干净音频路径配上文字标签。
- 划分：16,164 train、2,420 eval、1,416 test。
- 每条记录包含 noisy/clean 路径、来源、许可证、物化种子、声学测量、SFT 目标、DPO
  安全负样本和奖励上下文。
- Whisper embedding 缓存覆盖全部 20,000 条记录，训练重启时可直接复用。

## 训练链

训练设备为一张 32GB NVIDIA RTX 4080 SUPER。基座为 Qwen2.5-1.5B-Instruct，音频编码器
为 frozen Whisper-small；训练采用 QLoRA/NF4、BF16 计算、梯度累积和单卡 DeepSpeed
ZeRO-2。

| 阶段 | 步数 | 初始损失 | 最终损失 | 耗时 |
|---|---:|---:|---:|---:|
| SFT | 1,500 | 2.314612 | 0.087327 | 3,017.39s |
| DPO | 800 | 0.406933 | 0.325083 | 3,525.89s |
| GRPO | 300 | 1.3355e-5 | 4.6299e-5 | 9,689.05s |

三个阶段均完成并导出 adapter、`audio_projector.pt`、stage manifest 和保留 checkpoint。
关机后能够验证已完成阶段并跳过，只对未完成阶段从最近 checkpoint 恢复。

## 模型选择不是“最后阶段优先”

完整测试揭示了训练目标与部署质量之间的差异：

- SFT：1,416/1,416 条生成合法 JSON。
- DPO：抽检开始出现 `reduction_db: ?,?,?,?` 等非法数值占位符。
- GRPO：1,416/1,416 条 JSON 解析失败；平均 reward 为 0，1,200 个 group 全部饱和。

因此 4.0 发布 SFT adapter，而把 DPO/GRPO 保存为可审计的负结果。这不是绕开失败，而是
真正的模型选择：部署 checkpoint 必须通过独立结构校验和波形指标，不由训练阶段名称决定。

## 最终测试结果

SFT adapter 在全部 1,416 条 test 样本上进行确定性推理：

| 指标 | 结果 |
|---|---:|
| 合法 JSON | 1,416/1,416 |
| 推理失败 | 0 |
| batch size | 16 |
| 总耗时 | 843.95s |
| 吞吐 | 1.6778 条/s |
| 摊销平均延迟 | 595.97ms |
| p50 / p95 | 586.20 / 667.53ms |

修复 DSP 重建缺陷后，完整可执行泛化结果为：

| 指标 | 平均增益 |
|---|---:|
| SI-SDR | **+0.4509dB** |
| PESQ | **+0.04068** |
| STOI | **+0.002198** |
| 安全接受率 | **66.74%** |
| 安全回滚率 | **33.26%** |

`<5`、`5–10`、`10–20`、`>=20dB` 四个 SNR 桶的三项平均增益均为正。33.26% 的回滚
不是删除坏样本，而是控制器拒绝部署没有证明 SI-SDR 非负增益的候选波形。

DNSMOS、WER 和 speaker similarity 后端未配置，报告保留 `unavailable`，没有用常数或
伪造结果填充。

## 4.0 新增工程能力

1. 真实 noisy WAV 物化与数据许可证/来源清单。
2. Whisper 连续音频前缀，而不是只向 LLM 提供文本特征。
3. projector + LoRA 联合 SFT、DPO、GRPO。
4. DeepSpeed checkpoint 的非严格 module 冷恢复，兼容 NF4 量化元数据。
5. 已完成阶段验证、checkpoint 自动选择和安全续跑。
6. 缓存 embedding 的批量推理，吞吐从约 0.17 提升至 1.68 条/s。
7. 每批增量落盘、断点续推和原始非法响应取证。
8. DSP 动作白名单、数值边界、Nyquist 校验和执行后指标门。
9. 按数据集、语言、设备、噪声与 SNR 的泛化切片。
10. 流式、服务、鉴权和部署骨架。
11. 训练、预测、评测、环境、代码和发布资产的 machine-readable manifest。

## 声明边界

- 这是“语音增强控制策略”项目，不是领先的神经波形增强器。
- 平均增益为正但幅度有限，不能写成 SOTA。
- 数据只覆盖英文 LibriSpeech + ESC-50；多语言、重叠语音、真实会议室和设备迁移尚未证明。
- 测试安全门使用 clean reference；真实线上环境需要无参考质量估计或人工审核。
- 单卡 ZeRO-2 证明框架集成，不证明多卡扩展效率；ZeRO-3 offload 也不是单卡并行计算。
- DPO/GRPO 在本轮退化，不能写成“强化学习进一步提升效果”。

## 复现入口

```bash
export LSE_CONFIG=/root/autodl-tmp/lse-v4/configs/native_audio_autodl_32gb.json
bash scripts/autodl_v4_preflight.sh
bash scripts/autodl_v4_run.sh
```

完整步骤、目录、空间要求和断点恢复见
[`AUTODL_V4_RUNBOOK_ZH.md`](AUTODL_V4_RUNBOOK_ZH.md)。
