# LLM-Guided Speech Enhancement 2.0：新增内容与发布证据

发布日期：2026-08-02

版本：`v2.0.0`
硬件：单张 NVIDIA RTX 4090 24 GiB

## 1. 2.0 解决了什么问题

旧版脚本能分别启动 SFT 或 DPO，但数据格式、阶段来源、恢复状态和评测口径并未形成
一个可审计系统。2.0 把项目重构为“声学条件策略模型”：输入是音频路径及直接提取的
声学证据，输出是可执行 DSP 处方。模型负责决策，不冒充波形生成器。

## 2. 全量新增能力

### 2.1 数据工程

- 下载约 15 GB AISHELL 并校验 MD5；
- 从 40,000 个 clean 文件构造 120,000 条版本化退化合同；
- 生成 SFT、conservative DPO 和 GRPO 三套各 114,000 train / 6,000 holdout 数据；
- 每条记录保存源文件、采样率、时长、声学条件、目标处方、schema version 和来源；
- 物化 200 条真实噪声波形用于数据管线检查，其余明确标为参数化 proxy，不伪称
  120,000 个 noisy WAV。

### 2.2 三阶段连续后训练

- Qwen2.5-1.5B + LoRA + BF16；
- SFT 学会处方 JSON 与基础决策；
- conservative DPO 学习保守 chosen 相对过度处理 rejected 的偏好；
- GRPO/RLVR 对在线生成执行五分量程序 reward；
- pipeline 强制 SFT → cDPO → GRPO adapter 链，并保存可恢复状态和 stage manifest。

### 2.3 可验证 reward

五个独立分量是：格式、退化诊断、参数边界、动作—诊断一致性、过度处理约束。
reward 返回分数和 violations，能定位“为什么低分”。非法类型、数组冒充标量、NaN、
越界参数和矛盾动作均 fail closed。

### 2.4 DeepSpeed 和恢复能力

- 提供 ZeRO-2 与 ZeRO-3 CPU offload 配置；
- 最终 GRPO 使用 DeepSpeed 0.16.3 ZeRO-2 Engine；
- checkpoint-300 保存 stage/world-size 证据；
- OOM 后从完整 checkpoint-50 恢复，micro-batch 8→4、accumulation 2→4，保持有效
  batch 16、300 optimizer steps、256-token 上限和 policy/reference 双模型不变；
- 独立 NVML 记录峰值利用率 98%、峰值显存 24,067/24,564 MiB、峰值功耗 214.74 W。

单卡 DeepSpeed 证明 Engine、offload、checkpoint 和配置兼容，不证明 NCCL、多卡分片
或扩展效率。

### 2.5 公平阶段评测

Base、SFT、cDPO、GRPO 使用同一批 200 个 holdout ID、同一 prompt、解码和 reward。
有序 ID 列表 SHA-256 为
`50eaa2c3c59d1c5441757517fd9f9bc059f6b943ed55802b0e7fd82df8c75588`。

| 阶段 | JSON | 诊断 | 参数 | 一致性 | 过处理 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.355 | 0.222 | 0.000 | 0.268 | 0.355 | 0.222 |
| SFT | 1.000 | 1.000 | 1.000 | 0.820 | 1.000 | 0.964 |
| cDPO | 1.000 | 1.000 | 1.000 | 0.820 | 1.000 | 0.964 |
| GRPO | 1.000 | 1.000 | 1.000 | 0.820 | 1.000 | 0.964 |

持平是发布结论的一部分：当前同分布规则任务在 SFT 后饱和，不能声称 GRPO 显著优于
DPO。三阶段仍证明了偏好学习、在线 reward、reference、恢复与评测工程闭环。

### 2.6 工程化与发布

- 统一 JSON/YAML 配置、CLI override、固定 seed、`resume=auto` 和原子状态写入；
- AutoDL bootstrap/preflight/full runner；
- 45 项测试、Ruff lint/format 和 Python 3.10/3.11/3.12 CI；
- 公开最终 [GRPO LoRA](https://huggingface.co/jatshi/Audio-Codec-LLM-Qwen2.5-1.5B-GRPO-LoRA)、模型卡、训练摘要、阶段矩阵和哈希；
- 新增 README 动态演示、深度学习手册和完整事故复盘。

## 3. 真实训练结果

| 阶段 | Optimizer steps | Runtime | 核心指标 |
| --- | ---: | ---: | --- |
| SFT | 7,125 | 5,243.8 s | held-out loss 0.08168；token acc 0.96517 |
| cDPO | 7,125 | 10,633.3 s | pair acc 1.0；margin 2.188927 |
| GRPO | 300 | 4,247.43 s（稳定续训段） | mean reward 0.946019；181/300 步有非零组内方差 |

## 4. 2.0 中真正踩过的坑

1. 隐式 vanilla reference 导致极端 KL/loss 与 NaN；改为冻结、显式、同源 reference；
2. Qwen2.5 训练态 gradient checkpointing 让生成重复 `{`；拆分训练与生成状态并回归；
3. DPO 偏好 margin 饱和；重新校准 beta 并保留原始分布；
4. 合法 JSON 中非标量数值触发 reward 类型错误；在算术前做有限标量验证；
5. 罕见长 completion 在 micro-batch 8 OOM；保持有效 batch 的前提下降 micro-batch；
6. 训练结束同进程残留约 24 GiB，使评测准备 CPU offload；在全新进程加载最终 adapter。

每个事故的现象、证据、错误修法、最终修复与面试回答见
[从零手搓学习手册](audio_llm_v2_from_scratch_zh.md#19-20-真实实施日志与事故复盘)。

## 5. 面试与简历边界

可以说：

> 在单张 RTX 4090 上构建 Qwen2.5-1.5B LoRA 的 SFT→cDPO→GRPO/RLVR 连续后训练，
> 设计五分量可验证奖励、显式 reference、DeepSpeed ZeRO-2 checkpoint、OOM 等效 batch
> 恢复和同 ID 四阶段评测，并公开最终 adapter 与机器可读证据。

不能说：端到端恢复波形、120k 真实 noisy audio、多 GPU 分布式训练、GRPO 显著优于
DPO、或提升 PESQ/STOI/SI-SDR。当前版本没有这些证据。
