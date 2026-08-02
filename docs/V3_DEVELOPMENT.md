# Audio-Codec-LLM 3.0：原生音频条件、闭环增强与学习复盘

## 1. 问题引入

早期版本把音频先转成文字，再让 LLM 输出一份增强参数。这个方案容易演示，却丢失了音色、混响、噪声频谱和说话人特征；LLM 也只“开药方”，并不知道执行后音质是否真的改善。3.0 把它升级成可验证闭环：直接编码原始音频，将连续音频表示注入语言模型，生成有边界的动作，执行 DSP/神经增强，再测量结果，最后 accept、revise 或 rollback。

项目重点是音频理解、LLM 条件注入、多目标奖励、DeepSpeed 训练契约和闭环可靠性，不是控制理论。

## 2. 3.0 新增内容

- `WhisperAudioProjector`：冻结 Whisper encoder，将变长音频 hidden states 池化并映射到 LLM embedding 维度。
- 原生音频 smoke：真实加载 Whisper-small 与 Qwen2.5-1.5B-Instruct，训练 projector。
- `NumpyDSPExecutor`：有界增益、去直流和 spectral gate，动作越界直接拒绝。
- closed-loop controller：execute → measure → revise → re-measure → accept/rollback。
- 多目标快照：SI-SDR、STOI、PESQ、DNSMOS、WER、speaker similarity；依赖缺失时写 `unavailable`，不伪造数值。
- DeepSpeed ZeRO-2/ZeRO-3-offload 配置、单卡 contract smoke 与阶段矩阵。
- 真实模型权重 `audio_projector.pt`、环境 freeze 和 run manifest。

## 3. 音频怎样注入 LLM

Whisper encoder 输出 `H ∈ R^(T×d_audio)`。先根据 attention mask 做加权池化：

`h = sum(mask_t * H_t) / sum(mask_t)`。

再经过两层 projector：

`z = W2 * GELU(W1 * h + b1) + b2`。

如果要生成多个 audio prefix token，可以把输出维度设为 `K * d_llm` 后 reshape 成 `K × d_llm`。随后将这些向量拼到文本 token embedding 前面，并同步扩展 attention mask。训练初期冻结 Whisper 和 Qwen，只更新 projector，能显著降低显存与过拟合风险。

这里不能把 projector 叫“audio codec”。它是连续条件映射器，没有离散码本、码率或重建目标。项目名沿用历史，但文档必须讲清真实模块语义。

## 4. 闭环为什么优于一次性参数预测

一次性策略只能最小化预测动作与标签的差异，而闭环可以直接对结果负责。执行前后形成 metric snapshot，控制器根据硬约束与多目标得分判断：

- accept：关键指标改善且没有破坏说话人/可懂度约束；
- revise：有改善空间且动作预算未耗尽；
- rollback：整体退化或触发安全边界。

多目标不能简单追求 SI-SDR 最大。过强降噪可能提高波形指标却删除辅音，导致 WER 上升；也可能改变说话人 embedding。合理做法是先设硬门槛，再在可行解中优化加权目标或 Pareto 前沿。

## 5. 4090 实验结果

真实模型：Whisper-small + Qwen2.5-1.5B-Instruct。训练 2 个 optimizer steps，loss 从 3.31035 降到 2.70892，均值 3.00964；峰值显存 3859.58 MiB。训练权重位于 `artifacts/v3/smoke/native_audio/audio_projector.pt`，大小约 7.1MB。

这只能证明数据流、反向传播、优化器和权重保存都工作，不能声称 projector 已达到通用音频理解能力。两步 loss 下降是 smoke evidence，不是收敛实验。

## 6. DeepSpeed 到底验证了什么

ZeRO-2 分片 optimizer state 和 gradient；ZeRO-3 进一步分片 parameter，并可 offload 到 CPU/NVMe。单卡 world size=1 时没有跨卡通信，很多分片收益退化为 0，offload 甚至更慢。因此单卡可以验证：

- 配置文件能被正确解析；
- rank/world-size、seed 和 checkpoint 路径正确；
- 模型/优化器状态可保存恢复；
- 未来多卡启动接口已准备好。

但不能写“单卡实现分布式并行加速”。真正的扩展效率需要至少两卡测量 throughput、通信占比和显存峰值。

## 7. 踩坑复盘

### Windows CRLF 导致 shell 失败

项目在 Windows 生成后上传到 Linux，脚本出现 `/usr/bin/env: bash\r` 一类错误。最终在打包/上传阶段统一规范 `scripts/*.sh` 为 LF，并用 `.editorconfig`/Git attributes 防止复发。这个问题说明跨平台可复现不仅是 Python 依赖，也包括文本文件格式。

### 模型下载与缓存

首次模型下载远慢于训练本身。解决方式是把 `HF_HOME` 指向数据盘持久缓存、关闭不稳定的 Xet 路径并保留模型 revision。删除远端缓存前必须先确认源码、产物和需发布权重已转移。

### 显存占用不高是否浪费 4090

projector-only smoke 冻结了两个大模型，所以峰值只有约 3.86GB。4090 的价值在于快速真实前向/反向并保留后续全量/LoRA 训练空间；但如果长期只训练 7MB projector，较小显卡也足够。租卡选择应由峰值工作负载决定，而不是项目名称。

### 指标后端不能静默缺失

PESQ、STOI、DNSMOS 的依赖和采样率约束不同。缺包时若返回 0，会把“没有测”误当成“质量极差”。3.0 用 `unavailable + reason` 表示缺失，并阻止报告把它纳入均值。

## 8. 面试问答

**为什么冻结 Whisper？** smoke 阶段先验证条件映射是否可学，冻结 encoder 可减少变量和显存。数据规模扩大后再比较 frozen、LoRA 和部分解冻。

**为什么不用 ASR 文本代替音频？** 文本保留语义，但丢失噪声、房间、音色和韵律；增强动作恰恰依赖这些非文本信息。

**怎样证明 rollback 有用？** 构造会提高增益但恶化 WER/说话人相似度的动作，验证执行后快照触发 rollback，输出波形回到上一个可接受版本。

**loss 降了能说明效果吗？** 不能。loss 只证明训练目标被优化；最终要用留出集的增强质量、WER、说话人保持和失败率评估。

## 9. 亲手复现路线

1. 不看代码写出 masked mean pooling，并用 padding 前后结果一致性测试验证。
2. 手写一个两层 projector，把输出 reshape 为 4 个 LLM prefix token。
3. 冻结基座，只训练 projector，检查只有 projector 参数具有 gradient。
4. 实现一个故意过强的 spectral gate，证明闭环会 rollback。
5. 分别启动普通 PyTorch、DeepSpeed ZeRO-2 和 ZeRO-3 contract，比较单卡峰值与耗时，并解释为什么这不是多卡加速结论。

完成后，应能从张量形状、训练参数、增强指标和工程失败四个层面完整讲清项目。
