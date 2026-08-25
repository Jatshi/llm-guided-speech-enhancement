# LLM-Guided Speech Enhancement 4.0 学习与故障复盘

这份文档回答的不是“项目有哪些名词”，而是面试官沿着代码、数据、损失、训练、推理和故障
继续追问时，应该怎样把完整因果链讲清楚。

## 1. 项目到底解决什么问题

传统语音增强模型直接把 noisy waveform 映射成 clean waveform，优点是端到端，缺点是很难
解释为什么使用某种处理、参数是否安全、失败时是否应该保持原音。4.0 把任务拆成两层：

1. 多模态策略层读取音频与测量证据，生成结构化增强计划。
2. 确定性执行层验证计划、处理波形、重新测量并决定接受或回滚。

创新重点不是“让 LLM 代替 DSP”，而是让 LLM 做证据驱动的控制器，把执行权和安全边界留给
可验证程序。

## 2. 音频怎样进入语言模型

### 2.1 Whisper 不是 ASR 文本接口

系统调用 frozen Whisper-small encoder，获得形如 `[B, T, 768]` 的连续隐状态。它保留
语音内容、噪声、频谱和时间结构，比把音频先转写成文字再送入 LLM 信息更完整。

### 2.2 为什么压成 16 个 prefix token

Whisper 帧数较长，直接与文本拼接会造成上下文和显存开销。`AudioPrefixProjector` 用 mask
感知的压缩器把可变长度帧映射为 16 个 1,536 维向量，与 Qwen 的文本 embedding 同维。

```text
Whisper hidden [B,T,768]
  → mask-aware projector
  → audio prefix [B,16,1536]
  → concatenate(text embeddings)
  → Qwen causal LM
```

padding mask 很关键：没有 mask，补零帧会参与聚合，长度较短的音频会得到系统性偏差。

## 3. SFT、DPO、GRPO 分别训练什么

### 3.1 SFT

目标是让模型学习“证据 → 合法处方”的基本映射。对目标 token 做交叉熵：

```text
L_SFT = -Σ_t log πθ(y_t | audio, prompt, y_<t)
```

teacher forcing 允许目标序列所有位置并行计算，因此训练单步通常比逐 token 自回归推理更
容易吃满 GPU。本轮 SFT 从 2.3146 降到 0.0873，并在 test 上得到 100% JSON 有效率。

### 3.2 conservative DPO

DPO 使用 chosen/rejected 对，不显式训练 reward model。标准形式比较策略模型相对参考模型的
偏好对数比：

```text
z = β[(logπθ(y+|x)-logπθ(y-|x)) - (logπref(y+|x)-logπref(y-|x))]
L_DPO = -log σ(z)
```

label smoothing 给“chosen 必然正确”留出不确定性，避免偏好 margin 无限增大。本项目的负样本
是规则生成的过处理处方，不是人类偏好；规律过强会让 DPO 学到数据模板的捷径。

### 3.3 GRPO

GRPO 对同一输入生成多个候选，以组内 reward 均值和方差构造相对 advantage，再用裁剪策略
目标更新：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + ε)
L_policy = -min(ratio_i A_i, clip(ratio_i,1-ε,1+ε) A_i)
L = L_policy + β KL(πθ || πref)
```

如果同组所有候选 reward 相同，标准差接近 0，advantage 就失去学习信号。本轮出现 1,200 个
饱和 group、平均 reward 为 0；这不是“loss 很小所以训练很好”，而是 reward 链已经塌缩。

## 4. QLoRA 与 DeepSpeed 到底起什么作用

### QLoRA

- 基座权重以 NF4 4-bit 保存，减少显存。
- 前向时按块反量化到 BF16/FP16 参与矩阵乘法。
- LoRA 参数、梯度和优化器状态保持较高精度。
- 冻结的 4-bit 基座不直接做梯度更新；更新的是 LoRA 和 audio projector。

因此“量化训练”不是所有计算都用 4-bit。4-bit 主要是权重存储，累加与梯度仍需高精度。

### DeepSpeed ZeRO

本轮单卡使用 ZeRO-2，主要证明代码与 DeepSpeed optimizer/checkpoint 链兼容。单卡没有其他
rank 可以分片，所以不会获得真正的跨卡并行收益。ZeRO-3 offload 可以把状态挪到 CPU，降低
显存但增加 PCIe 传输；它也不等于“单卡并行计算”。

## 5. 为什么最初推理比想象中慢

旧实现对 1,416 条样本逐条执行：读取音频、重新运行 Whisper、batch=1 自回归生成最多 192
token，并在全部结束后才写文件。GPU 只有约 17% 利用率，每条约 5.95 秒。

修复方法：

1. 复用训练阶段已经保存的 Whisper embedding。
2. 对可变帧数 hidden/mask 做 batch padding。
3. 文本使用 left padding，避免 decoder-only 模型从 padding 位置开始生成。
4. batch size 提升到 16。
5. 每批写入 `predictions.partial.jsonl`，重启时按 sample ID 跳过已完成项。
6. 失败时保存 `raw_response`，使非法 JSON 可取证。

最终吞吐为 1.68 条/s，约提升 10 倍，并且关机不再意味着整轮推理归零。

## 6. 最重要的故障与修复

### 故障一：训练阶段已经完成，重启却要从头跑

根因：原 pipeline 只认命令流程，不验证阶段产物。修复后 `_completed_stage_report` 同时检查
stage manifest、artifact manifest、adapter 和 projector；完整则跳过，不完整才找 checkpoint。

面试表述：恢复依据必须是“可验证状态”，不能只看目录存在或日志中出现 `completed`。

### 故障二：DeepSpeed 冷恢复报 NF4 metadata unexpected keys

根因：bitsandbytes 的量化 metadata 会进入 state dict，而 DeepSpeed 默认严格 module load 对
版本差异过于敏感。修复为恢复 optimizer/ZeRO 状态时对 module 使用非严格加载，同时继续验证
关键训练资产，DPO 最终从 checkpoint-100 正确续跑。

### 故障三：AutoDL 预检反复阻止已缓存任务

首次运行需要为下载和物化预留 80GB，但续跑时模型、数据和 embedding 都已存在，仍按首次
阈值检查会浪费计费时间。使用 `LSE_MIN_FREE_GIB` 区分首次准备和缓存续跑；离线推理也不应因
Hugging Face 连通性失败而停止。

### 故障四：DPO/GRPO 输出非法占位符

现象是 `reduction_db: ?,?,?,?`。训练 loss 数值正常，但结构能力已经退化。最终 GRPO 1,416
条全部 JSON 失败。处理方式不是做字符串替换伪造数值，而是：

- 失败关闭并记录原始输出；
- 比较 SFT/DPO/GRPO checkpoint；
- 选择 1,416/1,416 合法的 SFT；
- 在模型卡中公开 DPO/GRPO 负结果。

更进一步的研究修复应包括 JSON grammar constrained decoding、格式 reward 的非饱和设计、
人工偏好数据、DPO 前后结构回归门以及每个阶段的独立 early stopping。

### 故障五：首轮 DSP 指标灾难性下降

首轮 SFT 输出结构全部合法，但平均 SI-SDR 增益达到异常的 `-40.76dB`。这提示问题不一定在
模型，应沿“处方 → DSP → 波形 → 指标”逐层定位。

根因是 `_stft` 使用 75% overlap，而 `_istft` 没传 `noverlap`，SciPy 使用默认 50%。分析和
合成时间网格不一致，即使频谱不修改也无法还原原波形。

新增往返测试先稳定复现：

```text
max |ISTFT(STFT(x)) - x| = 1.5368938   # 修复前
max |ISTFT(STFT(x)) - x| = 1.788e-7    # 修复后
```

修复后完整 1,416 条结果变为 SI-SDR +0.451dB、PESQ +0.0407、STOI +0.00220。这个案例很
适合回答“训练结果异常时如何定位”：先做组件不变量测试，不要立即重训模型。

### 故障六：Windows 脚本在 Linux unexpected EOF

4 个旧脚本使用 CRLF，Linux `bash -n` 报 `unexpected end of file`。增加 `.gitattributes`：

```gitattributes
*.sh text eol=lf
```

并规范化现有脚本。这个故障说明跨平台工程不能只靠 Python 测试，还要检查 shell 语法与行尾。

## 7. 安全门如何工作

生成的处方先通过 JSON、必需字段、动作白名单、有限数值、频率与 Nyquist 边界。执行后计算
candidate 相对 noisy input 的客观增益。当前 benchmark 要求 SI-SDR gain 不小于 0：

```text
if valid_plan and si_sdr_gain >= 0:
    output = candidate
else:
    output = noisy_input  # rollback
```

回滚率 33.26% 表示系统明确拒绝了 471 条未证明有效的候选，而不是偷偷丢掉这些样本。候选
指标仍进入报告，因此研究者可以看到策略真正的分布，而部署输出保持安全。

## 8. 面试时应怎样概括贡献

建议用下面的逻辑，不要只背技术栈：

> 我把原有文字证据 Demo 升级成真实音频条件的可验证控制系统。20,000 条物化 noisy WAV 经
> frozen Whisper 编码，由 projector 压成连续前缀，联合 QLoRA 训练 Qwen 的 SFT、DPO 和
> GRPO。模型只提出结构化 DSP 计划，执行器负责参数边界、波形处理、客观复测和安全回滚。
> 完整测试发现后训练阶段发生结构退化，所以系统没有盲选 GRPO，而是用 1,416 条测试证据选择
> SFT。最终 JSON 有效率 100%，SI-SDR/PESQ/STOI 平均增益均为正，66.74% 候选被接受，其余
> 回滚。过程中还定位并修复了 DeepSpeed/NF4 冷恢复、低吞吐推理和 STFT/ISTFT 重建缺陷。

这段话的核心是：数据真实、链路完整、指标可核验、失败不隐藏、选择有证据。
