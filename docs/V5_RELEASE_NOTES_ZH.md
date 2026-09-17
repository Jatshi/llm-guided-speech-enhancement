# v5.0 Prescription-MM-DiT 发布说明

## 发布结论

v5.0 将原先“LLM 生成增强处方 + DSP 执行”的控制系统扩展为可训练的波形增强链路：
处方、观测语音和流状态由 MM-DiT 联合建模；规划侧同时保留 LLM 的结构化动作输出，并
用声学路由器与安全门控约束何时可以执行。此次发布来自 10,000-step 正式训练和完整
200 样本六臂测试，不是 smoke 或只验证能运行的 demo。

## 新增内容

- 51,717,122 参数三流 MM-DiT：target、observed、prescription 在联合注意力中交互；
- observed-source residual rectified flow：从 noisy STFT 出发学习 clean residual；
- prescription dropout 与 mismatch no-op 目标：缺少或错配处方时学习保守不动作；
- Qwen2.5-1.5B LoRA 规划器：重新生成严格 JSON，测试集格式通过率 200/200；
- ExtraTrees 声学路由器：仅在 train 上拟合，test 诊断准确率 79%；
- validation-only 0.70 置信度阈值与动作能力门控；
- 逐样本 safety decision、原始输出、融合结果和回退原因的可审计记录；
- DeepFilterNet3、谱减、无处方、oracle、shuffled、predicted 六臂对照。

## 关键结果

最终 safety-gated predicted 分支取得 +0.114 dB SI-SDRi、+0.132 dB SNRi、2.027 PESQ、
0.881 STOI，回退率 76.5%。系统对 clean、reverb、telephone 以及低置信样本默认拒绝
执行。DeepFilterNet3 在非 clean 子集仍更强，因此 v5.0 的有效结论是“带能力边界的
选择性增强”，而不是“全条件超过强基线”。

## 失败经验与修复

1. 从高斯噪声直接生成完整 STFT，任务同时要求内容重建和处方控制，模型容易忽略处方；
   改为 observed-source residual flow 后，任务转化为局部修正。
2. 旧 LoRA 严格 JSON 为 0/200，修复括号后又坍缩到单一 white 类；重新建立与执行器一致
   的训练契约后，格式恢复到 200/200，但语义准确率只有 29.5%。
3. completion-only loss 没有解决数值特征映射，准确率反降到 20.5%；因此把底层声学分类
   交给可解释的 ExtraTrees，让 LLM 负责动作参数与解释。
4. 只使用置信度门控仍会处理少量高置信 clean；增加动作能力门控后，执行器不支持的
   dereverb/bandwidth-extension 条件直接拒绝，避免把“不会做”伪装成“增强失败”。

## 复现与发布资产

- 代码、配置、测试和文档：GitHub tag `v5.0.0`；
- 权重与模型卡：Hugging Face `jatshi/LSE-Prescription-MM-DiT-v5`；
- 机器可读评测：`validation/v5_mmdit/`；
- 完整运行记录：`MMDIT_AUTODL_EXECUTION_REPORT_2026-09-17.md`；
- 本地归档：F 盘 `artifacts/mmdit_autodl_20260917/`，含原始 tar、解包产物和校验值。
