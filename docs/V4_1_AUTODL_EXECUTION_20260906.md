# 2026-09-06 AutoDL GRPO 修复执行记录

任务跨多个克隆实例续跑；本文不保存临时 SSH 地址或密码。

GPU 实测为 NVIDIA GeForce RTX 4080 SUPER 32760 MiB。开始时显存占用 1 MiB；数据盘 130 GiB，约 82 GiB 可用。

部署到独立目录 `/root/autodl-tmp/lse-v4-1`；旧项目 `/root/autodl-tmp/lse-v4` 及其 SFT/DPO/GRPO 输出保留。
新项目的 `data` 和 `outputs/native_v4` 链接到旧项目相应目录。新输出使用 `outputs/native_v4_1_grpo_canary` 和 `outputs/native_v4_1_grpo`。

盘点发现：SFT adapter/projector、20000 条 materialized 音频与 native manifest 均保留；此前为缩减数据盘删除了虚拟环境、基础模型和 Whisper embedding cache。因而需要重新安装环境、下载两个基础模型、补算缓存。

部署时补充修复：

- 移除 shell 中对已有 embedding index 的强制要求，允许 recovery 自动补算缺失缓存。
- 模型下载仅取 safetensors、JSON、tokenizer 等所需文件，避免下载重复 PyTorch/TensorFlow/ONNX 权重。
- GRPO 生成启用 KV cache；训练前向仍不缓存。
- 短训限制为 128 条 train 样本，先通过 GPU 验证再编码完整数据集。
- 增加 canary 每组、GRPO 每步、音频缓存阶段日志。

运行路径：

- 环境：`/root/autodl-tmp/lse-v4-1-env`
- 基础模型：`/root/autodl-tmp/lse-v4-models`
- 安装日志：`/root/autodl-tmp/lse-v4-1-logs/setup-retry.log`
- 模型下载日志：`/root/autodl-tmp/lse-v4-1-logs/models.log`
- 临时安装文件：`/root/autodl-tmp/lse-v4-1-tmp`（避免填满系统盘）

## 2026-09-08：12GB 实例续跑

克隆实例的 GPU 实测为 RTX 3080 Ti 12GB，因此新增两个独立配置：

- `configs/native_audio_autodl_grpo_recovery_canary_12gb.json`
- `configs/native_audio_autodl_grpo_recovery_12gb.json`

第一次以 4 候选、512 tokens、无梯度检查点运行，在第 4 个反向更新前 OOM；错误现场显示显存已用约 11.37GB，另需申请 806MB。第二次把候选数降为 3，虽不再 OOM，但只有 1/8 组出现奖励差异，被 `non_saturated_group_rate >= 0.2` 门禁在训练前拒绝。这证明不能靠缩小 GRPO 组组来绕过奖励门禁.

最终低显存策略保留 4 候选，将训练序列设为 384 tokens、生成上限 160 tokens，并启用 `expandable_segments` 与非重入梯度检查点。后者修复了重入检查点触发 DeepSpeed ZeRO-2 对同一参数重复规约的断言错误。

12 步短训真实结果：

| 指标 | 结果 |
| --- | ---: |
| 状态 | completed |
| steps | 12 |
| mean reward | 0.970312 |
| valid JSON rate | 0.989583 |
| non-saturated group rate | 0.25 |
| mean anchor loss | 0.109329 |
| mean policy loss | 2.73e-8 |
| mean KL | 1.25e-6 |
| elapsed | 664.87 s |

第一次正式全量门禁暴露了 reward 粒度不足：48 个生成虽然全部为有效 JSON，但只有
2/12 组产生非饱和相对奖励，因此流程在第 0 个 optimizer step 前按设计中止。没有降低
20% 门槛，而是在现有参数合法性分量内加入相对于 `expected_response` 的连续参数校准。
目标来自已知的合成退化配置，不使用 LLM 裁判；合法但偏离目标的增益、衰减和频率参数
得到连续降分。新增回归测试先复现“偏离参数仍满分”，再验证修复。

修复后的正式门禁达到 48/48 有效 JSON、12/12 非饱和组。随后在 RTX 3080 Ti 12GB 上
完成 300 steps：训练期 sampled JSON=0.991458、非饱和组=0.998333、最长连续饱和仅 1 组，
没有生成 `collapse_report.json`。完整测试集 1,416/1,416 推理成功，最终晋升门禁通过：
SFT mean reward 0.953821，GRPO mean reward 0.976320，placeholder rate 均为 0。

详细指标、声明边界和文件路径见 `docs/V4_1_RESULTS_ZH.md`。
