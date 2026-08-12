# LLM-Guided Speech Enhancement 3.0 深度学习、踩坑与面试手册

> 目标：能从张量形状、音频条件注入、闭环优化、多目标指标、DeepSpeed 边界和工程失败
> 六个层面讲清项目，不把两步 smoke 说成模型收敛。

## 1. 2.0 与 3.0 的核心区别

2.0 把声学证据写成文本/JSON，让 LLM 生成结构化增强处方。优点是可解释、容易做 SFT/
DPO/GRPO；缺点是 ASR 文本丢失音色、噪声频谱、混响和韵律，且一次性处方不知道执行
后是否改善。

3.0 增加两条链：原始音频 → Whisper hidden states → LLM prefix；处方 → DSP 执行 →
指标复测 → accept/revise/rollback。

## 2. 音频 prefix 的张量过程

Whisper encoder 输出 `H ∈ R^(B×T×d_audio)`，mask 为 `M ∈ {0,1}^(B×T)`：

`h_b = sum_t(M_bt * H_bt) / max(sum_t M_bt, 1)`。

两层 projector：

`z = W2 GELU(W1 h + b1) + b2`。

若使用 `K` 个 prefix token，输出维度设为 `K*d_llm`，再 reshape 为
`B×K×d_llm`。将其拼到文本 embedding 前，同时在 attention mask 前补 `K` 个 1，在
label 前补 `K` 个 `-100`，使 prefix 参与条件计算但不承担 token 交叉熵。

## 3. 为什么先冻结 Whisper 和 Qwen

项目首先要证明 projector 的维度、mask、gradient 和保存路径正确。冻结两端只训练中间层：

- 显存低，迭代快；
- 参数因果清晰，能检查只有 projector 有 gradient；
- 小数据下不容易破坏基座；
- 失败时更容易定位是数据/拼接还是全模型优化。

后续才比较 Whisper LoRA、Qwen LoRA、部分解冻与全量微调。峰值显存 3.86GB 不表示
4090 无用，只说明当前 smoke 是 projector-only。

## 4. 为什么它不是 audio codec

真正 codec 通常包含 encoder、离散/连续瓶颈、码率和 decoder/reconstruction objective。
本项目 projector 没有码本、码率或波形重建，只是把音频表示映射到 LLM embedding。

正确说法是“native audio-conditioned policy LLM”或“连续音频前缀投影器”。面试时主动
澄清，反而能体现对模型边界的理解。

## 5. 闭环怎样做决策

一次性模型输出动作 `a_0`，执行得到音频 `y_1`，测量快照 `m_1`。控制器比较执行前后：

- 硬约束：WER 不能恶化超过门槛、speaker similarity 不能跌破安全线；
- 软目标：SI-SDR/STOI/PESQ/DNSMOS 尽量提高；
- 动作预算：修订次数、增益和 gate 强度有限。

满足硬约束且总体改善则 accept；有改善空间且预算足够则 revise；触发安全边界或总体退化
则 rollback 到上一版本波形。

## 6. 指标不能混成一个“音质分”

- SI-SDR：波形失真与干扰抑制；
- STOI：可懂度代理；
- PESQ/DNSMOS：感知质量代理；
- WER：下游 ASR 可懂性；
- speaker similarity：身份/音色保持。

过强降噪可能提高 SI-SDR，却删除辅音并使 WER 上升。建议先设硬约束，再在可行区域做
加权或 Pareto 选择。依赖缺失必须标 `unavailable`，返回 0 会把“没测”误作“极差”。

## 7. 两步训练结果怎么解释

Whisper-small + Qwen2.5-1.5B 真实前向/反向，loss 3.31035 → 2.70892，导出 7,089,880
字节 projector。它证明：音频读取、encoder、masked pooling、prefix 拼接、loss、optimizer
和保存链路能运行。

它不证明：留出集泛化、增强质量提升、WER 下降或主观听感改善。简历应写“完成真实模型
链路验证与可加载 projector”，不能写“音质提升 X%”。

## 8. DeepSpeed ZeRO 怎样理解

- ZeRO-1 分片 optimizer state；
- ZeRO-2 再分片 gradient；
- ZeRO-3 再分片 parameter；
- offload 用 CPU/NVMe 换 GPU 显存，通常牺牲吞吐。

单卡 world size=1 没有跨卡分片收益。可以验证配置解析、rank、seed、checkpoint 和恢复，
不能声称分布式加速。多卡才需要测 tokens/s、通信占比、扩展效率和每卡峰值。

## 9. 关键踩坑案例

### Windows CRLF 让 Linux shell 失效

脚本出现 `/usr/bin/env: bash\r`。修复是在 `.editorconfig`/Git 中固定 shell 为 LF，并在
上传前执行语法检查。跨平台复现不仅是 Python 包，也包括换行符和可执行位。

### 下载时间比训练长

解决方式是 GPU 开机前准备代码与 manifest，`HF_HOME` 指向数据盘，固定 revision，缓存
复用；清理缓存前先把产物、权重和日志全部拉回并做 SHA-256。

### 指标包缺失被错误写成 0

0 是有效测量值，不能表示 unavailable。项目改用 tagged result：`value` 或
`unavailable(reason)`，聚合器只对真实 value 求均值。

## 10. 高频面试问答

**为什么不用 ASR 文本？** 文本保留语义，但丢失噪声、房间、音色和韵律；增强动作依赖
这些非文本因素。

**为什么 LLM 适合输出增强策略？** 它能融合多种证据并生成结构化动作和理由；真正的
安全来自 schema、参数边界、执行器和复测 gate，而不是相信自然语言。

**怎样证明 rollback 有效？** 构造提高响度但恶化 WER/说话人相似度的动作，验证快照
触发 rollback 且输出恢复到上一个可接受版本。

**显存这么低为何用 4090？** 当前只训练 projector，小卡也够；4090 价值在于快速真实
链路验证和后续 LoRA/部分解冻余量。租卡应由最大实验决定，不由项目名字决定。

## 11. 亲手练习

- 手写 masked mean，验证 padding 前后结果一致；
- 构造 4 个 prefix token，并正确扩展 mask/label；
- 检查只有 projector 参数有 gradient；
- 故意产生越界 DSP action，确认 fail closed；
- 构造指标互相冲突的案例并触发 rollback；
- 分别解释 ZeRO-2、ZeRO-3 和单卡退化。

2.0 的 SFT/DPO/GRPO、奖励与数据管线基础继续参考
[2.0 发布说明](V2_RELEASE_NOTES.md) 和
[2.0 从零手搓学习手册](audio_llm_v2_from_scratch_zh.md)。
