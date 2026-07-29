# LLM-Guided Speech Enhancement 2.0：从零手搓完整学习手册

版本：2.0  
目标读者：需要真正理解并能独立重建、调试、答辩本项目的人  
基座模型：`Qwen/Qwen2.5-1.5B-Instruct`  
训练主线：LoRA + SFT → DPO → GRPO/RLVR  
验证硬件：单张 RTX 4090 24GB

---

## 0. 先把项目说准确

### 0.1 一句话定义

这是一个**声学条件驱动的语音增强策略语言模型**：系统读取带来源追踪的音频记录和声学退化属性，输出结构化的噪声诊断、保守的增强动作、参数、理由与置信度，并用可执行规则对这些处方做偏好优化和强化学习。

### 0.2 它不是哪三种东西

它不是：

1. 直接把 noisy waveform 变成 clean waveform 的 Demucs/FullSubNet；
2. 能听懂任意音频 token 的原生 Omni 模型；
3. 已经完成主观听感提升验证的商用增强器。

当前 Qwen2.5-1.5B 是文本语言模型。训练输入是由真实音频路径、受控退化过程和结构化声学属性转写成的 prompt；输出是增强**处方**，不是波形。仓库保留直接音频路径和特征接口，后续可以接 Audio Encoder 或真正的 DSP 执行器，但在这些链路实测前不能写成“端到端语音增强模型”。

### 0.3 为什么这个问题值得做

传统增强系统经常只有固定配置：

```text
所有样本 → 同一个降噪强度 → 同一个滤波器 → 输出
```

真实场景却同时变化：

- 噪声类型不同：white、pink、HVAC、cafe；
- SNR 不同：干净语音不该被重度抑制；
- 混响不同：有混响时仅做 spectral subtraction 不够；
- 带宽不同：电话带宽和全带宽语音的处理策略不同；
- 风险不同：过度降噪会损害可懂度和音色。

本项目把问题重写成：

\[
\text{acoustic evidence}
\longrightarrow
\text{diagnosis}
\longrightarrow
\text{bounded enhancement prescription}
\]

语言模型负责组合策略，确定性 reward 负责约束安全边界。这能展示数据工程、LoRA、SFT、DPO、GRPO、RLVR、DeepSpeed、评测和可复现发布，而不是只展示一次 API 调用。

### 0.4 学完后的闭卷验收

你应该能不看文档回答：

- 为什么训练顺序不能写成三个互不相干的 base-model run？
- SNR 的定义是什么，如何把目标 SNR 转成噪声缩放系数？
- SFT、DPO、GRPO 的监督信号分别是什么？
- DPO 的 `beta` 控制什么？
- GRPO 为什么适合可验证 JSON 处方？
- 五个 reward 分量分别防什么错误？
- LoRA 的 `r`、`alpha` 和 target modules 如何影响训练？
- 为什么单卡 ZeRO-3 不能叫多 GPU 分布式训练？
- 为什么 reference reward 不是模型评测？
- 为什么 120,000 条 metadata 不等于 120,000 个 materialized noisy waveform？

---

## 1. 完整系统心智模型

### 1.1 数据流

```text
AISHELL-1 clean WAV
  ↓ 受控退化与来源记录
audio_manifest.v2.jsonl
  ├─ clean/noisy path
  ├─ noise type / SNR / RT60 / bandlimit
  ├─ source URL / license / MD5 / seed
  └─ target diagnosis + bounded actions
  ↓ 确定性转换
SFT messages
DPO prompt + chosen/rejected
GRPO prompt + reward_context
  ↓
Qwen2.5-1.5B + LoRA
  ↓ SFT
结构和任务冷启动
  ↓ DPO
偏好正确处方，压制过度处理
  ↓ GRPO
程序化 reward 优化
  ↓
冻结留出集生成 prediction
  ↓
format / diagnosis / bounds / consistency / overprocessing
```

### 1.2 三类核心对象

第一类是 `lse.audio_manifest.v2`。它是事实来源：

```json
{
  "schema_version": "lse.audio_manifest.v2",
  "sample_id": "000000_0",
  "audio": {
    "clean_path": "...wav",
    "noisy_path": "...wav",
    "source_role": "materialized_noisy_audio",
    "sample_rate": 16000
  },
  "acoustics": {
    "noise_type": "white",
    "snr_db": 5.5,
    "reverb_rt60": 0.21,
    "bandlimit_hz": null,
    "features": {}
  },
  "target": {
    "diagnosis": {
      "noise_type": "white",
      "reverb": true,
      "band_limited": false
    },
    "actions": [
      {
        "type": "spectral_subtraction",
        "reduction_db": 12.0,
        "low_hz": 80,
        "high_hz": 7600
      }
    ],
    "rationale": "Apply conservative suppression derived from measured degradation.",
    "confidence": 0.75
  },
  "provenance": {
    "dataset": "AISHELL-1",
    "license": "Apache-2.0",
    "source_url": "https://www.openslr.org/resources/33/data_aishell.tgz",
    "source_md5": "2f494334227864a8a8fec932999db9d8",
    "sampling_seed": 42
  }
}
```

第二类是 alignment records：

- `lse.sft.v2`：完整 messages，最后一条是 gold assistant；
- `lse.dpo.v2`：prompt、chosen、rejected；
- `lse.grpo.v2`：prompt、reward_context。

第三类是训练与评测证据：

- `stage_manifest.json`；
- `trainer_state.json`；
- TensorBoard event；
- `pipeline_status.json`；
- `predictions.jsonl`；
- `reward_report.json`；
- adapter 和 SHA-256。

### 1.3 为什么 schema version 必须写在每一行

如果只靠文件名区分格式，文件被复制、拼接或重命名后就可能被错误读取。每行带 schema 后：

- SFT Trainer 不会误读 DPO 行；
- 缺字段会在训练前失败；
- 旧版数据不会静默混入；
- 评测脚本能验证输入语义；
- 数据飞轮能追踪转换版本。

`read_jsonl` 对非法 JSON、非 object 行都会抛出带行号错误，绝不静默跳过坏数据。

---

## 2. 声学退化从公式到代码

### 2.1 SNR

信噪比定义：

\[
\operatorname{SNR}_{dB}
=10\log_{10}\frac{P_s}{P_n}
\]

其中：

\[
P_s=\operatorname{mean}(s^2),\qquad
P_n=\operatorname{mean}(n^2)
\]

如果已有原始噪声 \(n\)，希望得到目标 SNR \(r\)，缩放系数为：

\[
\alpha=
\sqrt{
\frac{P_s}
{P_n\cdot 10^{r/10}}
}
\]

最终：

\[
x=s+\alpha n
\]

仓库 `Degradation.add_noise` 正是这个实现，并给噪声功率加 `1e-12`，防止全零噪声除零。

### 2.2 四种噪声不是四个名字

- white：各频率近似等功率；
- pink：功率随频率升高而下降，更接近自然环境低频占优；
- HVAC：稳定低频和机械背景；
- cafe：模拟人声与环境混合的非平稳噪声。

训练时 `noise_type` 既进入 prompt，也进入 diagnosis reward。模型如果把 HVAC 诊断成 white，不会因为 JSON 合法就拿满分。

### 2.3 简化混响

教学退化使用指数衰减 impulse response：

\[
h[t]=e^{-t/\tau}
\]

归一化后与语音卷积，再做 dry/wet 混合：

\[
y=0.6x+0.4(x*h)
\]

`RT60` 表示混响衰减 60 dB 所需时间。代码中的构造是受控数据增强近似，不是实测房间脉冲响应。它适合验证流程，不足以代表真实房间泛化。

### 2.4 带宽限制

`bandlimit` 使用四阶 Butterworth band-pass：

\[
y=\operatorname{filtfilt}(b,a,x)
\]

`filtfilt` 前后向滤波，能避免普通 IIR 的相位延迟。归一化截止频率必须除以 Nyquist 频率 \(f_s/2\)。

### 2.5 过度处理为什么是核心风险

增强不是“参数越大越好”。例如：

- `reduction_db > 18` 开始受罚；
- 总 reduction 超过 30 dB 继续受罚；
- `|gain_db| > 18` 被视为极端增益；
- high-pass 截止超过 180 Hz 可能损害男声基频；
- SNR ≥ 20 dB 时仍做超过 12 dB 抑制属于干净信号过处理。

这些边界不是医学或工业安全认证，而是可审计的工程约束。它们的价值是防止模型通过“所有动作都拉满”骗取表面诊断一致性。

---

## 3. 数据集是怎样构建的

### 3.1 本轮真实来源

本轮使用 OpenSLR SLR33 的 AISHELL-1：

| 项目 | 值 |
|---|---|
| license | Apache-2.0 |
| source archive | `data_aishell.tgz` |
| verified MD5 | `2f494334227864a8a8fec932999db9d8` |
| sampling seed | 42 |
| sampled clean files | 40,000 |
| degradations per file | 3 |
| manifest rows | 120,000 |
| materialized noisy WAV | 200 |

这组数字必须一起说。120,000 行表示受控声学条件与训练记录数量；为控制磁盘，只保存了 200 个 noisy waveform 作为直接音频检查，其余行保存 clean path 和可复现退化 metadata。不能把 120,000 行写成“发布了 120,000 个 noisy WAV”。

### 3.2 本轮 alignment 数量

确定性转换后：

| 阶段 | train | eval |
|---|---:|---:|
| SFT | 114,000 | 6,000 |
| DPO | 114,000 | 6,000 |
| GRPO | 114,000 | 6,000 |

split 由 seed 控制。每个源 record 会派生三种训练合同，sample ID 和 provenance 保持一致。

### 3.3 SFT 样本

形式：

```json
{
  "schema_version": "lse.sft.v2",
  "sample_id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "noise=white; snr=5.5; rt60=0.21; ..."},
    {"role": "assistant", "content": "{\"diagnosis\":...,\"actions\":...}"}
  ]
}
```

SFT 让模型先学会任务语言和严格输出结构。如果跳过 SFT，DPO/GRPO 会把大量计算浪费在“怎样输出 JSON”而不是策略差异。

### 3.4 DPO 样本

形式：

```json
{
  "schema_version": "lse.dpo.v2",
  "prompt": [...],
  "chosen": [{"role": "assistant", "content": "...保守且一致的处方..."}],
  "rejected": [{"role": "assistant", "content": "...过度处理或逻辑冲突..."}],
  "preference": {
    "reason": "chosen is evidence-consistent; rejected over-processes"
  }
}
```

`corrupt_target` 不是随机乱码，而是制造能教会模型偏好的负例，例如过强 reduction、矛盾参数或错误诊断。

### 3.5 GRPO 样本

```json
{
  "schema_version": "lse.grpo.v2",
  "prompt": [...],
  "reward_context": {
    "noise_type": "white",
    "snr_db": 5.5,
    "reverb_rt60": 0.21,
    "bandlimit_hz": null,
    "expected_response": "..."
  }
}
```

GRPO 不把 expected response 当作逐 token 标签。Trainer 对每个 prompt 采样多个 completion，再调用 `grpo_reward` 独立判分。

---

## 4. LoRA：为什么一张 4090 能训练

### 4.1 基本公式

冻结原权重 \(W\)，只学习低秩更新：

\[
W'=W+\Delta W,\qquad
\Delta W=\frac{\alpha}{r}BA
\]

其中：

- \(A\in\mathbb{R}^{r\times d_{in}}\)；
- \(B\in\mathbb{R}^{d_{out}\times r}\)；
- \(r=16\)；
- \(\alpha=32\)；
- scaling 为 \(\alpha/r=2\)。

### 4.2 target modules

本项目同时覆盖 attention 与 MLP：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

只训练 q/v 更省参数，但结构化策略和理由生成也依赖 MLP 表达能力，所以这里使用更完整的 target modules。

### 4.3 为什么使用 BF16

BF16 与 FP32 有相同的 8-bit exponent，动态范围更大，通常比 FP16 更不容易 overflow；mantissa 较短，但训练大模型通常足够。RTX 4090 支持 BF16。

### 4.4 gradient checkpointing

不保存所有中间 activation，而是在 backward 时重算：

\[
\text{更低显存}
\longleftrightarrow
\text{更多计算}
\]

SFT 与 cDPO 使用 `use_reentrant=False`，兼容现代 PyTorch/Transformers 的推荐路径。

GRPO 是一个必须单独验证的例外。Qwen2.5 + Transformers 4.48 在训练态启用
gradient checkpointing 后，会把生成阶段的 `use_cache` 关闭。4090 实跑中，
同一个 cDPO adapter 在普通采样时能稳定生成合法 JSON，但进入该路径后会连续生成
`{"{"{"...` 直至 256 token 上限，导致所有 reward 都变成 0。关闭 GRPO 的
checkpointing 后，首步平均 completion 从 256 token 恢复到约 90 token，
reward 从 0 恢复到 0.96875，单步时间也从 46.9 秒降到 8.8 秒；峰值 allocated
显存为 15.33 GiB。因此本项目保留 SFT/cDPO checkpointing，只对 GRPO 关闭。
这不是为了省略训练，而是修复生成语义并提高真实吞吐。

### 4.5 有效 batch size

单卡时：

\[
B_{effective}
=B_{device}\times A_{grad}
\]

SFT：

\[
8\times2=16
\]

DPO：

\[
4\times4=16
\]

GRPO：

\[
4\times4=16
\]

GRPO 的 per-device batch 还必须被 `num_generations=2` 整除，否则组内采样无法正确组织，训练前会直接失败。

---

## 5. SFT：先让模型会说正确的“语言”

### 5.1 目标函数

\[
\mathcal{L}_{SFT}
=-\sum_t\log\pi_\theta(y_t\mid x,y_{<t})
\]

训练只更新 LoRA 参数，base Qwen 权重保持冻结。

### 5.2 本轮关键配置

| 参数 | 值 | 解释 |
|---|---:|---|
| epochs | 1 | 全量 114k 一遍 |
| batch | 8 | 4090 单步样本数 |
| grad accumulation | 2 | 有效 batch 16 |
| LR | 2e-4 | LoRA SFT 常见量级 |
| max length | 1024 | 覆盖 prompt 与 JSON |
| warmup ratio | 0.03 | 减轻起步震荡 |
| packing | false | 不把多个样本拼成同序列 |

### 5.3 为什么实测后从 batch 2 调到 8

最初 `batch=2, accumulation=8` 虽然有效 batch 仍为 16，但 GPU 只使用约 7.9GB，吞吐低，预计超过 5 小时。实测确认显存余量后改为 `batch=8, accumulation=2`：

- 有效 batch 不变；
- 数据顺序和目标不变；
- 显存提高到约 18.9GB；
- GPU 利用率和功耗明显提高；
- SFT 预计时间缩短到约 1.4 小时。

这体现了一个重要工程知识：gradient accumulation 能模拟大 batch 的优化步，但小 micro-batch 无法充分利用 GPU 的矩阵并行度。

---

## 6. DPO：从“会输出”到“偏好保守正确”

### 6.1 目标函数

\[
\mathcal{L}_{DPO}
=-\log\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
\]

\(y_w\) 是 chosen，\(y_l\) 是 rejected。

### 6.2 `ref_model=None` 不等于参考策略一定正确

PEFT/TRL 可以使用 adapter-disabled 的同一 base 作为隐式 reference。这只在
“当前 adapter 是从该 base 新建的增量”时等价于初始策略。本项目的 GRPO 却是从
已经训练完成的 cDPO adapter 继续优化；关闭 adapter 得到的是原始 Qwen，而不是
GRPO 的 step-0 策略。

TRL 0.16 的隐式 reference 在本次实跑第 1 步产生约 \(5\times10^{18}\) 的 KL、
约 \(2\times10^{17}\) 的 loss 和 NaN 梯度，随后触发 CUDA 概率张量断言。修复方式
不是把 `beta` 改成 0，而是额外加载一份相同 cDPO adapter：

- policy copy：可训练；
- reference copy：`requires_grad=False` 且 `eval()`；
- DeepSpeed ZeRO-2 下 reference 以 stage-0 engine 常驻单卡；
- `beta=0.04` 保持不变；
- `reference_policy.json` 记录来源、模式和生成配置。

修复后首步 KL 降到正常的 \(10^{-4}\) 量级，且没有取消 KL 约束。面试时应明确：
单卡上第二份 reference 会增加显存，但它保证了“相对 cDPO 初始策略”的数学语义。

### 6.3 从 `beta=0.1` 校准到 `beta=0.01`

`beta` 控制相对 reference 的偏好强度。过大可能让已经分离的 chosen/rejected 很快进入 sigmoid 饱和区，过小则偏好约束不足。4090 全量实跑在 `beta=0.1` 时观测到初始原始对数概率差约为 140，对应 DPO logit 约 14、loss 约为 0、梯度仅约 \(10^{-6}\)，说明偏好目标在训练开始前已经饱和。

依据实测 margin，本轮将 `beta` 校准为 `0.01`。此时初始 DPO logit 约为 \(0.01\times140=1.4\)，既保留 chosen 优于 rejected 的方向，又使 sigmoid 处于仍有有效梯度的区域。

标准 DPO 在 455 步时又把 rejected log-prob 推到约 -692，margin 达到 6.85、loss 降到约 0.001，说明程序生成的单向偏好标签会诱导策略继续增大已经足够的间隔。因此最终训练加入 `label_smoothing=0.1`，使用 conservative DPO：

\[
\mathcal{L}_{cDPO}=-(1-\epsilon)\log\sigma(z)-\epsilon\log\sigma(-z),
\quad \epsilon=0.1.
\]

其有限最优 logit 为

\[
z^*=\log\frac{1-\epsilon}{\epsilon}=\log 9\approx2.20,
\]

从而避免 rejected 概率被无边界地推向零。该校准没有减少样本、epoch 或有效 batch，而是修正训练目标的过置信行为。本轮 DPO 使用：

- LR `5e-6`；
- batch `8`；
- accumulation `2`；
- label smoothing `0.1`；
- 1 epoch；
- max prompt `768`；
- max total length `1024`。

DPO 同时计算 chosen/rejected，显存和计算通常高于同 batch 的 SFT，因此 micro-batch 比 SFT 小。

---

## 7. GRPO/RLVR：让程序化约束直接进入优化

### 7.1 为什么是 GRPO

PPO 往往需要 value model；GRPO 用同组样本相对 reward 构造 advantage，减少额外 critic 成本：

\[
A_i=\frac{r_i-\mu_r}{\sigma_r+\epsilon}
\]

本项目的任务存在强可验证结构，非常适合 RLVR：

- JSON 是否合法；
- noise type 是否一致；
- 参数是否越界；
- 动作是否符合声学条件；
- 是否过度处理。

### 7.2 本轮关键配置

| 参数 | 值 |
|---|---:|
| max steps | 300 |
| batch | 4 |
| accumulation | 4 |
| num generations | 2 |
| max completion | 256 |
| temperature | 0.7 |
| beta | 0.04 |
| LR | 1e-5 |
| save steps | 50 |

temperature 不能为零，否则同组 completion 过于相似，reward 标准差接近零，组内相对学习信号变弱。

### 7.3 五个 reward 分量

最终 reward 是等权平均：

\[
R=\frac{
R_f+R_d+R_b+R_c+R_o
}{5}
\]

其中：

#### Format

必需字段：

```text
diagnosis, actions, rationale, confidence
```

缺一个字段扣 0.25。diagnosis 必须是 object，actions 必须是非空 list，confidence 必须位于 `[0,1]`。

#### Diagnosis

noise type 完全相等得 1；包含关系得 0.75；否则 0。若 RT60 表示有混响，而模型写 `reverb=false`，再乘 0.75。

#### Parameter bounds

逐项检查：

| 参数 | 合法范围 |
|---|---|
| reduction_db | 0–24 |
| gain_db | -24–6 |
| q | 0.5–20 |
| low_hz | 0–20,000 |
| high_hz | 20–24,000 |

同时要求 `low_hz < high_hz`。最后返回 passed/checks，而不是简单 0/1，给局部改进留下梯度方向。

#### Consistency

- 有混响应包含 dereverb/WPE 类动作；
- 有带宽限制应包含 bandwidth extension/EQ；
- high-pass 截止过高会和语音频段冲突；
- 有噪声却没有 action type 会扣分。

#### Overprocessing

起始为 1，再按过强 reduction、极端 gain、过高 high-pass、累计抑制和干净语音重处理扣分，最低为 0。

### 7.4 为什么 reward 必须返回 violations

只保存一个 `0.6` 无法调试。violations 会告诉你：

```text
invalid_json
noise_type_mismatch
reverb_without_dereverb_action
action_0_reduction_db_out_of_range
clean_signal_overprocessed
```

这既服务训练故障分析，也能形成 hard-case 数据飞轮。

---

## 8. DeepSpeed：单卡实测与正确表述

### 8.1 ZeRO 分什么

- Stage 1：optimizer states；
- Stage 2：再加 gradients；
- Stage 3：再加 parameters。

真正的“分片”需要多个 data-parallel rank。单卡 `WORLD_SIZE=1` 没有其他 GPU 可分，因此不能声称完成多 GPU 分布式训练。

### 8.2 单卡为什么仍有价值

单卡可以验证：

- DeepSpeed Engine 能否与 TRL/PEFT 共存；
- BF16、gradient accumulation 是否一致；
- checkpoint 是否能保存和恢复；
- CPU optimizer/parameter offload 是否工作；
- 显存与吞吐的真实交换关系；
- 配置能否迁移到未来多卡环境。

### 8.3 本项目的 fail-closed 合同

配置里带 `_portfolio_contract`：

- declared world size 必须等于实际 `WORLD_SIZE`；
- BF16 与 FP16 不能同时开启；
- Stage 2 不允许 parameter offload；
- offload device 只能是 CPU/NVMe；
- single-GPU-only 配置不能被多卡 launcher 误用。

这是为了阻止“配置文件里写了 ZeRO-3，所以我有多卡经验”的简历夸大。

### 8.4 单进程环境变量

直接从 Python pipeline 进入 DeepSpeed 时，仍需：

```text
MASTER_ADDR
MASTER_PORT
RANK
LOCAL_RANK
WORLD_SIZE
```

本项目在确认 world size 为 1 后自动补齐，并写入 manifest。它不是伪造多卡，而是满足 DeepSpeed singleton process-group 初始化。

---

## 9. 训练流水线为什么可恢复

### 9.1 状态机

`run_pipeline`：

```text
write status=running
for stage in [sft,dpo,grpo]:
    validate dataset
    resolve DeepSpeed
    find checkpoint
    train stage
    verify final adapter
    atomically update status
generate final predictions
score predictions
write status=complete
```

任一异常会写：

```json
{
  "status": "failed",
  "failed_stage": "dpo",
  "error": "RuntimeError: ..."
}
```

### 9.2 `resume=auto`

它寻找形如 `checkpoint-500` 的最新有效目录。恢复的是 optimizer/scheduler/trainer state，不只是加载 adapter 后重新开始计数。

### 9.3 为什么每阶段还要保存 `final/`

checkpoint 服务中断恢复；`final/` 服务阶段串联和发布。DPO 启动前要求 SFT final adapter，GRPO 启动前要求 DPO final adapter。这样不会意外从 base model 重跑。

### 9.4 原子写状态

先写临时文件，再替换正式 JSON。机器断电时最多保留旧状态，不会留下半截 JSON 让自动化误判成功。

---

## 10. 离线评测和消融

### 10.1 两种完全不同的评测

`evaluate_rewards(dataset, predictions_path=None)` 使用 reference expected response，作用是验证 reward 实现和数据合同。它不是模型成绩。

正式评测必须：

1. 加载 base + final GRPO adapter；
2. 对留出 prompt 做 deterministic generation；
3. 保存 `sample_id + response`；
4. 再调用相同 reward；
5. 报告缺失 prediction 数量。

### 10.2 核心指标

- `valid_json_rate`：可被解析为 JSON 的 prediction 占比，评测器将
  `RewardBreakdown.valid_json` 转为 0/1 后在实际匹配到的 prediction 上求均值；
- format mean；
- diagnosis mean；
- parameter bounds mean；
- consistency mean；
- overprocessing mean；
- total reward mean；
- violation frequency；
- generation latency；
- peak VRAM；
- 分声学条件表现。

### 10.3 消融怎么做

至少比较：

- 去掉 format reward；
- 去掉 diagnosis reward；
- 去掉 bounds reward；
- 去掉 consistency reward；
- 去掉 overprocessing reward。

同一批 prediction 可以离线重算不同权重，回答“评价函数如何变化”；但若要回答“训练时去掉某 reward 会怎样”，必须重新训练对应 GRPO 变体，不能把离线重算冒充训练消融。

### 10.4 PESQ/STOI/SI-SDR 为什么不能伪造

当前模型输出处方，不直接输出 enhanced waveform，因此没有波形对就无法计算这些指标。正确报告应写 `not_run` 和原因，而不是填零或把 reward 当成 PESQ。

只有在接入 DSP executor 或神经增强模型并生成 waveform 后，才可以计算：

\[
\text{SI-SDR}
=10\log_{10}
\frac{\|\alpha s\|^2}
{\|\alpha s-\hat{s}\|^2}
\]

以及 STOI、PESQ、DNSMOS。

---

## 11. 源码阅读路线

### 第一遍：只读合同

```text
lse_v2/contracts.py
lse_v2/io.py
tests/test_contracts.py
```

目标：能画出四种 schema，理解为什么错误行 fail fast。

### 第二遍：读数据生成

```text
src/degradation.py
src/generate_degraded_data.py
lse_v2/data_cli.py
```

目标：能推导 SNR scale，解释 provenance、seed 和 materialized/noisy proxy。

### 第三遍：读 reward

```text
lse_v2/rewards.py
tests/test_rewards.py
```

目标：闭眼写出五个分量和每个边界。

### 第四遍：读训练

```text
lse_v2/training.py
lse_v2/config.py
```

目标：解释 base/adapter 如何串联，SFT/DPO/GRPO 各自 Trainer 输入是什么。

### 第五遍：读 DeepSpeed

```text
lse_v2/deepspeed.py
configs/deepspeed/
tests/test_deepspeed.py
```

目标：能区分 Engine 集成、offload 和多 GPU 分片。

### 第六遍：读 pipeline 与评测

```text
lse_v2/pipeline.py
lse_v2/inference.py
lse_v2/evaluation.py
```

目标：能说明 resume、atomic status、真实 prediction 与 reference validation 的区别。

---

## 12. 从空目录重建：十个 commit

1. `chore: create package and test skeleton`
2. `feat: define versioned audio and alignment contracts`
3. `feat: add deterministic acoustic degradation and provenance`
4. `feat: derive sft dpo grpo datasets`
5. `feat: implement verifiable prescription rewards`
6. `feat: add lora sft trainer`
7. `feat: chain dpo from sft adapter`
8. `feat: chain grpo with rlvr rewards`
9. `feat: add deepspeed contracts resume and manifests`
10. `feat: add heldout inference evaluation and release assets`

每个 commit 都应先有失败测试，再写最小实现。这样你不是背现成代码，而是在重演设计决策。

---

## 13. 七天实操学习计划

### Day 1：数据和声学

- 手算三组 SNR scale；
- 用 NumPy 生成 white/pink noise；
- 画 clean/noisy waveform 与 spectrogram；
- 修改 RT60 并听差异。

验收：能解释为什么功率比用 10log10，幅度比用 20log10。

### Day 2：合同与数据转换

- 手写最小 audio record validator；
- 构造一个 SFT、DPO、GRPO row；
- 故意删字段，确认程序失败；
- 用相同 seed 构建两次并比 SHA-256。

### Day 3：reward

- 不看源码实现五个分量；
- 为每个分量写正反例；
- 设计一个 reward hacking 输出；
- 修改边界并观察消融。

### Day 4：LoRA 与 SFT

- 计算 target modules 的 LoRA 参数量；
- 跑 20 step SFT；
- 查看 loss、token accuracy、显存；
- 比较 batch 2/8 在有效 batch 相同时的吞吐。

### Day 5：DPO

- 构造三组 chosen/rejected；
- 推导 DPO log-ratio；
- 改 beta 做短跑；
- 检查 chosen reward 和 rejected reward 趋势。

### Day 6：GRPO

- 对同一 prompt 采样两个 completion；
- 手算 reward 和 group advantage；
- 制造全相同 reward，解释为什么梯度弱；
- 检查原始 completion 和 violations。

### Day 7：Pipeline、DeepSpeed 与答辩

- 中断训练并 auto resume；
- 对比 plain/ZeRO-2/ZeRO-3 offload；
- 从 final adapter 生成 prediction；
- 闭卷画完整架构；
- 完成后面的面试题。

---

## 14. 高频故障排查

### 14.1 `mpi4py` / distributed initialization

先看：

```bash
env | grep -E 'MASTER|RANK|WORLD_SIZE'
```

单进程直接进入 DeepSpeed 时必须有 singleton distributed env。不要为了绕过错误关闭 DeepSpeed后继续声称已验证。

### 14.2 `--local_rank` 不识别

DeepSpeed launcher 会注入 `--local_rank=0`。CLI 应同时接受下划线和连字符形式，并有测试覆盖。

### 14.3 CUDA OOM

按顺序处理：

1. 看 OOM 发生在模型加载、forward、backward 还是 save；
2. 降 micro-batch；
3. 等比例提高 accumulation，保持有效 batch；
4. 确认 gradient checkpointing；
5. 检查是否误加载第二份 reference model；
6. 再考虑 ZeRO offload。

不要第一反应缩短完整数据或取消 DPO/GRPO。

### 14.4 GPU 利用率低但显存也低

通常是 micro-batch 太小、数据预处理瓶颈或频繁 logging。保持有效 batch 不变，逐步提高 micro-batch，记录吞吐和 peak VRAM。

### 14.5 GRPO reward 全相同

检查：

- temperature 是否为零；
- completions 是否真的不同；
- reward_context 是否逐样本对齐；
- parsing 是否把所有输出都判成 invalid；
- reward 是否饱和；
- `num_generations` 与 batch 是否兼容。

本次实跑同时遇到了两种“全相同”，处理方式不同：

1. reward 全为 0：打印原始 completion 后发现是 checkpointing 生成退化，关闭
   GRPO checkpointing 后修复；
2. reward 都约为 0.95–1.0 且组内标准差为 0：输出合法但两次采样完全相同，
   advantage 仍为 0，需要提高探索温度。

固定 32 个 prompt 的校准结果表明：

| temperature | 合法 JSON | 文本不同采样对 | 非零 reward 差采样对 | 平均 reward |
|---:|---:|---:|---:|---:|
| 1.65 | 93.75% | 34.38% | 15.63% | 0.9000 |
| 1.70 | 90.63% | 37.50% | 15.63% | 0.8695 |
| 1.75 | 89.06% | 40.63% | 25.00% | 0.8617 |
| **1.80** | **87.50%** | **50.00%** | **34.38%** | **0.8414** |
| 1.85 | 79.69% | 50.00% | 37.50% | 0.7695 |
| 1.90 | 71.88% | 71.88% | 50.00% | 0.6938 |

最终选 `temperature=1.8, top_p=1.0`，因为它已经提供稳定的组内 reward 信号，
又没有像更高温度那样快速破坏 JSON。完整原始校准数字见
`docs/grpo_calibration.json`。

### 14.6 offline score 很高但输出不可用

抽查 raw prediction。程序 reward 可能存在漏洞，尤其是：

- 只检查字段存在，不检查类型；
- expected substring 可被否定句包含；
- 参数合法但动作语义无关；
- 引用/理由复制 prompt；
- 干净样本总是使用同一保守模板。

---

## 15. 面试问题与参考要点

### Q1：这个项目为什么叫 Audio-LLM？

因为模型的任务和训练信号来自音频退化条件，输出用于语音增强策略。但当前基座是文本 Qwen，声学条件以结构化 evidence prompt 输入，所以准确表述是 acoustic-conditioned policy LLM，不是原生 audio-token LLM。

### Q2：为什么不直接训练增强网络？

增强网络更适合生成 waveform；本项目关注策略选择、参数约束和可验证后训练，展示 SFT/DPO/GRPO 与工程闭环。未来可把处方接入 DSP/神经增强 executor，用真实音质指标反哺 reward。

### Q3：DPO 和 SFT 最大区别？

SFT 学唯一 gold token 序列；DPO 学 chosen 相对 rejected 的偏好，更适合表达“保守参数优于过度处理”。

### Q4：GRPO 比 DPO 多了什么？

DPO 依赖预先构造的偏好对；GRPO 可以对模型现场生成的多个答案执行 reward，让模型探索 gold 之外但仍满足约束的策略。

### Q5：为什么 reward 不用大模型裁判？

核心合同可程序验证。确定性 reward 成本低、可复现、能单测，也避免裁判模型偏差。主观听感未来可以作为独立人工或 DNSMOS 信号，不能替代结构约束。

### Q6：为什么诊断正确还不能满分？

因为动作可能越界、与诊断矛盾或过度处理。项目优化的是完整处方，不是分类器。

### Q7：单卡 ZeRO-3 有什么意义？

验证 Engine、CPU offload、checkpoint 和未来多卡配置，并量化显存/吞吐权衡；它不证明跨 GPU 分片。

### Q8：数据量最大的误导风险是什么？

把 120k metadata 说成 120k materialized noisy WAV，或把受控合成退化说成真实噪声语料。正确说法必须同时给出 40k clean、每文件三种条件、120k manifest、200 noisy audio checks。

### Q9：为什么最终评测必须加载 GRPO adapter 重新生成？

reference expected response 的满分只证明 reward 与数据相容；只有模型真实 prediction 才能证明训练结果。

### Q10：下一版最重要的升级是什么？

接入真正的 audio encoder 或 DSP executor，生成 enhanced waveform；建立 source-level train/validation/test；加入真实噪声/RIR；用 PESQ/STOI/SI-SDR/DNSMOS 和人工听测构成多目标 reward。

---

## 16. 简历可写与不可写

完成真实产物后可以写：

> 构建基于 Qwen2.5-1.5B + LoRA 的声学条件策略模型，打通 120k 版本化训练记录上的 SFT→DPO→GRPO/RLVR 流水线；设计 JSON、诊断、参数边界、一致性和过度处理五分量可验证奖励，并在单张 RTX 4090 上完成 DeepSpeed ZeRO-2 Engine 训练、断点恢复与留出集评测。

不能写：

- “训练了端到端语音增强大模型”；
- “120k 真实 noisy audio”；
- “完成多 GPU 分布式训练”；
- “提升 PESQ/STOI”，除非已有 waveform 对和真实报告；
- “GRPO 显著优于 DPO”，除非同一 test split 有阶段对照。

---

## 17. 最终自测

闭卷完成以下任务：

1. 画出 manifest → SFT → DPO → GRPO → prediction → reward report；
2. 推导目标 SNR 的噪声缩放系数；
3. 写一个最小 `score_prescription`；
4. 解释五个 reward 各自独立的必要性；
5. 解释 DPO `beta` 与 GRPO `beta` 的作用；
6. 计算三阶段有效 batch；
7. 说明 LoRA target modules；
8. 说出 pipeline 如何避免三阶段从 base 重跑；
9. 解释 reference validation 与 model evaluation；
10. 用一句话准确说明单卡 DeepSpeed 的证据边界。

如果这十项能全部现场完成，才算真正“像自己手搓出来的一样”掌握了项目。

---

## 18. 本轮真实实验结果

本节只接受从最终 `trainer_state.json`、`stage_manifest.json`、
`predictions.jsonl` 和 `reward_report.json` 自动汇总的数字。训练完成后
将记录 SFT/DPO/GRPO runtime、loss/reward、显存、留出集五分量、消融和
最终 Hugging Face adapter 链接；任何尚未产生的指标都不提前填造。
