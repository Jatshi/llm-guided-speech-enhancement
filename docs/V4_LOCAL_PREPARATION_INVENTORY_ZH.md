# LLM-Guided Speech Enhancement 4.0 本地准备工作统计与资产清单

> 统计日期：2026-08-25
>
> 文档状态：动态盘点；最终发布和 F 盘归档完成后需更新为交付版
> 安全说明：本文不记录 SSH 密码、GitHub Token、Hugging Face Token 或其他密钥

## 1. 路径结论

### 本地源码与准备工作的主目录

```text
C:\Users\jat_s\Documents\Codex\2026-07-17\https-talent-antgroup-com-campus-position\work\llm-guided-speech-enhancement
```

该目录是 4.0 开发、调试和发布准备的本地事实源，包含源码、测试、AutoDL
脚本、训练配置、运行手册、模型卡和 README。后续 GitHub 发布应从这里进行，而不是直接把
AutoDL 上的临时目录当作源码仓库。

### F 盘最终归档目标

```text
F:\博士内容归档\GitHub开源项目补充\llm-guided-speech-enhancement\v4.0-archive
```

父目录已经存在，但本轮 4.0 端到端归档尚未完成。最终归档会采用独立的 `v4.0-archive`
目录，不覆盖现有旧版本文件。

### AutoDL 当前运行路径

```text
代码：/root/autodl-tmp/lse-v4
环境：/root/autodl-tmp/lse-v4-env
模型：/root/autodl-tmp/lse-v4-models
数据：/root/autodl-tmp/lse-v4-data
结果：/root/autodl-tmp/lse-v4/outputs/native_v4
日志：/root/autodl-tmp/lse-v4-logs/full_run.log
```

## 2. 本地仓库总体统计

统计时排除了 `.git`、`.pytest_cache`、`.ruff_cache` 和 `__pycache__`。

| 指标 | 数值 |
|---|---:|
| 文件总数 | 166 |
| 文件总字节数 | 2,334,555 B（约 2.23 MiB） |
| Git 当前分支 | `codex/production-hardening` |
| 当前基线提交 | `a2a4ca57624a957445dbf4d475c116d57e638800` |
| Git 已跟踪文件 | 83 |
| 未跟踪文件 | 45 |
| 工作区变更项 | 52 |

工作区变更是 4.0 生产化补强内容，目前没有为了“看起来干净”而丢弃或覆盖；发布前需要在完整
测试、指标核验和文档审计后形成正式提交。

## 3. 本地目录分布

| 顶层区域 | 文件数 | 大小 | 主要用途 |
|---|---:|---:|---|
| `lse_v2/` | 30 | 255,536 B | 训练、推理、DSP、评测、服务和流式处理核心代码 |
| `tests/` | 27 | 67,265 B | 单元测试、契约测试、端到端与生产 DSP 测试 |
| `scripts/` | 24 | 54,636 B | AutoDL、DeepSpeed、数据准备、评测与发布脚本 |
| `docs/` | 15 | 101,539 B | 架构、运行手册、数据协议、盘点和版本学习文档 |
| `configs/` | 8 | 12,952 B | AutoDL、原生音频、DeepSpeed 与 smoke 配置 |
| `src/` | 9 | 24,132 B | Demo/应用层代码 |
| `assets/` | 2 | 1,380,920 B | README 和演示媒体资产 |
| `build/` | 30 | 238,864 B | 本地构建产物，可由源码重建 |
| `dist/` | 2 | 149,393 B | Python 发布包，可由源码重建 |
| `.light/` | 2 | 3,656 B | 项目状态与交接台账 |
| 根目录文档与配置 | 8 | 约 39 KiB | README、模型卡、计划、变更日志和依赖声明 |

## 4. 4.0 已准备的核心资产

### 训练与模型代码

- `lse_v2/native_training_pipeline.py`：Whisper 音频条件前缀、SFT、DPO、GRPO、QLoRA、
  DeepSpeed 与断点恢复。
- `lse_v2/native_inference.py`：真实音频推理、LoRA 加载、缓存 embedding 批量生成。
- `lse_v2/native_predict.py`：测试集推理、增量落盘、断点续推、延迟和吞吐统计。
- `lse_v2/native_alignment.py`：音频—语言对齐损失。
- `lse_v2/rewards.py`：处方结构、安全性与偏好/策略优化奖励。

### 可执行增强与评测代码

- `lse_v2/dsp.py`：参数校验、动作白名单、可执行 DSP 和失败关闭边界。
- `lse_v2/end_to_end.py`：处方执行、指标验证和安全回滚。
- `lse_v2/generalization.py`：按数据集、语言、设备、噪声和 SNR 分桶评测。
- `lse_v2/metrics.py`：SI-SDR、PESQ、STOI 等指标。
- `lse_v2/service.py`、`lse_v2/streaming.py`：服务和流式处理骨架。

### AutoDL 与分布式训练配置

- `configs/native_audio_autodl_32gb.json`：本轮 32GB GPU 完整训练配置。
- `configs/native_audio_autodl_canary.json`：小规模金丝雀验证配置。
- `configs/native_audio_4090.json`：4090 配置。
- `configs/deepspeed/ds_zero2.json`：本轮实际使用的 ZeRO-2 配置。
- `configs/deepspeed/ds_zero3_offload.json`：ZeRO-3 CPU offload 实验配置。
- `scripts/autodl_v4_bootstrap.sh`、`autodl_v4_preflight.sh`、
  `autodl_v4_prepare_data.sh`、`autodl_v4_run.sh`、`autodl_v4_public_full_run.sh`：
  从环境准备到完整运行的脚本链。

### 文档

- `docs/AUTODL_V4_RUNBOOK_ZH.md`：AutoDL 运行手册。
- `docs/DATA_AND_ANNOTATION_PROTOCOL_ZH.md`：数据与标注协议。
- `docs/PRODUCTION_HARDENING.md`：生产化补强说明。
- `docs/RUN_MANIFEST_V4_LOCAL.md`：本地运行清单。
- `README.md`、`MODEL_CARD.md`、`run_manifest.md`：对外说明、模型卡和复现入口。

### 测试

本地共有 27 个测试文件，覆盖训练目标、音频条件策略、DeepSpeed 契约、原生数据、处方解析、
生产 DSP、端到端增强、泛化评测、服务和流式链路。2026-08-25 新增了以下回归保护：

- 缓存批量推理的索引校验、非法 JSON 取证和分批逻辑测试。
- STFT → ISTFT 波形往返一致性测试。
- 修复前该测试最大误差为 `1.5368938`；修复后为 `1.7881393e-7`。

## 5. AutoDL 训练与产物统计

### 三阶段训练

| 阶段 | 步数 | 初始损失 | 最终损失 | 平均损失 | 状态 |
|---|---:|---:|---:|---:|---|
| SFT | 1,500 | 2.314612 | 0.087327 | 0.128559 | 已完成 |
| DPO | 800 | 0.406933 | 0.325083 | 0.344742 | 已完成 |
| GRPO | 300 | 1.3355e-5 | 4.6299e-5 | 1.5874e-5 | 已完成，但奖励塌缩 |

GRPO 的平均奖励为 `0.0`，1,200 个 group 全部饱和。训练流程在工程意义上完整执行并保存，
但模型质量不能仅凭“流程跑完”宣称成功。

### 适配器与 checkpoint

| 资产 | 统计 |
|---|---|
| SFT final | 11 个文件，96,988,730 B |
| DPO final | 11 个文件，96,984,783 B |
| GRPO final | 11 个文件，96,984,805 B |
| 每阶段 `audio_projector.pt` | 44,117,192 B |
| SFT checkpoint | `checkpoint-1300`、`checkpoint-1400` |
| DPO checkpoint | `checkpoint-600`、`checkpoint-700` |
| GRPO checkpoint | `checkpoint-200`、`checkpoint-250` |

### 评测状态

| 评测 | 结果/状态 |
|---|---|
| GRPO 最终适配器 JSON 有效率 | 0/1,416；全部安全失败，不能发布为最佳模型 |
| SFT 适配器 JSON 有效率 | 1,416/1,416；失败 0 |
| SFT 缓存批量推理 | 843.95 秒；1.68 条/秒；摊销均值 595.97ms |
| 首轮 SFT DSP 泛化 | 已完成，但受 STFT/ISTFT 重叠率缺陷污染，保留作故障证据 |
| 修复后 SFT DSP 泛化 | 正在运行；完成后才形成最终可发布指标 |

首轮负指标不是静默删除的数据。其审计文件会保留在
`outputs/native_v4/sft_generalization`，修复后结果写入独立的
`outputs/native_v4/sft_generalization_stft_fixed`，从而保留故障—修复的证据链。

### 远端存储

| 项目 | 当前规模 |
|---|---:|
| `outputs/native_v4` 总量 | 26,439,133,682 B（约 24.62 GiB） |
| 音频 embedding 缓存 | 约 8.7 GiB，20,000 条，全部可复用 |
| 完整运行日志 | 249,279 B |
| 原始 GRPO 测试/泛化报告 | 已生成并保留 |
| SFT 推理结果 | 约 2.48 MB，包含 canonical、partial 和报告文件 |

## 6. 当前质量结论

| 能力 | 状态 | 说明 |
|---|---|---|
| 三阶段训练可恢复执行 | 已验证 | 关机后阶段 final 与 checkpoint 均保留，完成阶段可跳过 |
| DeepSpeed/QLoRA | 已验证 | 单卡使用 ZeRO-2；ZeRO-3 offload 配置已提供 |
| 原生音频测试推理 | 已验证 | 真实测试集 1,416 条均执行 |
| 增量与断点续推 | 已验证 | 每批写入 partial 文件，可跳过已完成 sample ID |
| 最终阶段自动择优 | 部分完成 | 已确认 GRPO 退化、SFT 结构有效；等待修复后客观指标 |
| DSP 安全回滚 | 已验证 | 负增益候选不会冒充最终增强音频 |
| 泛化指标 | 运行中 | STFT/ISTFT 源头缺陷已修复并重新评测 |
| GitHub 发布 | 待完成 | 需在最终指标与 README 审核后推送 4.0 |
| Hugging Face 发布 | 待完成 | 只上传经完整评测选出的适配器和诚实模型卡 |
| F 盘归档 | 待完成 | 远端产物、日志、清单和校验值尚未全部传输 |

## 7. 最终归档与发布分工

### GitHub 保存

- 源码、配置、测试、脚本和文档。
- 小型 JSON/CSV 指标与发布清单。
- README 中的 4.0 架构、真实指标、复现命令和 Hugging Face 链接。
- 不提交原始数据集、大型 checkpoint、音频缓存和密钥。

### Hugging Face 保存

- 完整评测后选出的最佳 LoRA adapter。
- 对应的 `audio_projector.pt`、adapter 配置、模型卡和指标摘要。
- 明确写出基础模型、Whisper 版本、许可证、限制与失败阶段，不把 GRPO 退化隐去。

### F 盘保存

- 本地源码快照和远端源码快照。
- `outputs/native_v4` 中三阶段 final、保留 checkpoint、run manifest、预测和泛化报告。
- 完整日志、环境冻结、发布清单和 SHA-256 文件。
- 可公开重新下载的大型原始数据/基础模型可以不重复归档，但必须记录来源、版本与重建命令。

## 8. 完成交付前仍需执行

1. 等待修复后 1,416 条泛化评测完成并核验全部分桶指标。
2. 生成 SFT/DPO/GRPO 模型选择报告，指定可发布适配器。
3. 运行完整 pytest、Ruff、脚本语法和 artifact manifest 校验。
4. 更新 README、模型卡、4.0 变更说明、故障复盘与复现命令。
5. 提交并推送 GitHub 4.0 源码。
6. 上传 Hugging Face adapter 与模型卡并回填链接。
7. 建立 F 盘独立归档，比较远端/本地文件数和字节数，对关键文件做 SHA-256 校验。
8. 生成最终交付报告后再停止心跳监控。

## 9. 统计口径与更新规则

- 文件数量与大小来自 2026-08-25 的本地和 AutoDL 实际文件系统，不是计划估算。
- `build/`、`dist/` 属于可重建产物，计入本地占用但不作为源码规模。
- 远端 `sft_generalization_stft_fixed` 正在增长，因此其当前文件数和字节数不写成最终值。
- 最终版必须补充 GitHub/Hugging Face 链接、F 盘目录、传输统计和校验结论。
