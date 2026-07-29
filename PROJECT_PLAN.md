# LLM-Guided Speech Enhancement 2.0实施计划

## 一、目标

将旧版“结构化声学描述 → SFT/DPO → DSP建议”原型升级为单张RTX 4090可复现的
音频证据驱动后训练项目，完整覆盖：

1. 真实音频或声学特征的数据入口；
2. SFT、DPO、GRPO三阶段后训练；
3. 可验证奖励与奖励消融；
4. 断点续训、日志、配置、产物追踪；
5. 模型预测级离线评测；
6. AutoDL一键预检、安装和运行。
7. 单GPU DeepSpeed ZeRO-2/ZeRO-3 offload运行与可核验性能对照。

## 二、技术边界

### 已纳入2.0

- 音频文件路径和可直接提取的声学特征进入`lse.audio_manifest.v2`。
- Qwen2.5-1.5B读取序列化声学证据并生成结构化DSP处方。
- LoRA SFT、DPO、GRPO顺序训练。
- 规则奖励可独立于训练器离线复算。
- 最终adapter生成eval预测并产生机器可读报告。

### 不在2.0中冒充完成

- Qwen本身没有新增音频encoder，不能称为端到端原始波形AudioLM。
- 未开GPU前没有训练曲线、checkpoint和模型提升结论。
- 未接真实语音增强波形模型前，不报告PESQ/STOI增益。

## 三、工作包与验收门

### WP1：数据契约

交付：

- `lse.audio_manifest.v2`
- `lse.sft.v2`
- `lse.dpo.v2`
- `lse.grpo.v2`
- legacy metadata迁移器
- 可选真实音频特征提取器

验收：

- 重复`sample_id`立即失败；
- 缺字段、非法采样率、错误split立即失败；
- 固定seed得到相同划分；
- train/eval按样本ID隔离；
- `--check-audio-files`能阻止悬空路径。

### WP2：SFT

目标：学习稳定输出合法的诊断、动作、理由和置信度JSON。

默认配置：

- Qwen2.5-1.5B-Instruct
- BF16 LoRA r=16
- max length 1024
- batch 8 × gradient accumulation 2（有效batch 16）
- 1 epoch

GPU验收：

- `outputs/v2/sft/final/adapter_config.json`存在；
- `stage_manifest.json.status=complete`；
- TensorBoard日志存在；
- 至少保留一次eval loss；
- 随机抽查20条生成结果，记录JSON合法率。

### WP3：DPO

目标：让模型偏好证据一致、参数安全的处方，拒绝过度抑制与自信幻觉。

数据：

- chosen来自结构化目标；
- rejected由确定性安全扰动生成；
- 每对记录偏好来源、原因和margin；
- 后续人工复核对必须使用新的`preference.source`。

全量初跑若在`beta=0.1`下出现loss≈0和极小梯度，则依据实测原始对数概率差校准
`beta`。本轮在margin约140时采用`beta=0.01`，使初始DPO logit约为1.4；同时采用
`label_smoothing=0.1`的conservative DPO，把有限最优logit约束在
\(\log((1-0.1)/0.1)\approx2.20\)，避免程序生成负例被持续推向负无穷。数据量、
epoch和有效batch均保持不变。

GPU验收：

- SFT adapter作为可训练policy初始值；
- `ref_model=None`使用PEFT参考策略，避免重复加载；
- DPO adapter与状态清单存在；
- chosen平均奖励高于rejected；
- 保存至少20条失败偏好对供人工检查。

### WP4：GRPO

目标：以自动验证信号优化格式、诊断、参数安全、一致性和不过处理。

默认约束：

- 1.5B模型；
- 每提示2个候选；
- max completion 256；
- beta 0.04；
- 最多300步作为第一轮；
- 不默认使用7B，避免单4090 rollout显存风险。

GPU验收：

- 首先运行20步短实验；
- 显存峰值低于23.5GB；
- 无NaN、OOM和奖励全常数；
- 五个奖励分量均有方差；
- 通过后再恢复至300步；
- 保存最终adapter、日志与状态。

### WP5：离线评测与消融

交付：

- 最终模型eval预测JSONL；
- 平均总奖励；
- 五个分量均值；
- 违规类型样例；
- 分别移除每项奖励后的重算结果。

验收：

- 预测与`sample_id`一一对应；
- 缺预测时报告明确指出回退到reference；
- reference自检不得标记为模型benchmark；
- 所有报告含schema版本与生成时间。

### WP6：AutoDL

统一接口：

```bash
scripts/autodl_v2_preflight.sh
scripts/autodl_v2_bootstrap.sh
scripts/autodl_v2_run.sh
```

模式：

- `PORTFOLIO_V2_MODE=smoke`：CPU离线校验；
- `PORTFOLIO_V2_MODE=full`：4090完整训练。

验收：

- preflight检查Python、配置、CUDA、显存、磁盘；
- bootstrap创建隔离venv并`pip check`；
- run在数据缺失时停止，而不是启动空训练；
- 任意阶段中断后再次运行能自动选择最大编号checkpoint；
- `pipeline_status.json`能定位失败阶段。

### WP7：DeepSpeed

交付：

- `configs/deepspeed/ds_zero2.json`
- `configs/deepspeed/ds_zero3_offload.json`
- `scripts/deepspeed_single_gpu_smoke.sh`
- `scripts/distributed_contract_smoke.py`
- `docs/deepspeed_comparison_template.csv`

证据边界：

- 两个配置固定声明`world_size=1`；
- ZeRO-2单卡没有跨GPU状态分片收益；
- ZeRO-3 CPU offload是显存换速度，不预设一定更优；
- CPU多进程smoke不运行GPU collective，不可声称已完成多GPU测试；
- 真正多GPU运行需新建与真实`WORLD_SIZE`一致的配置并在对应硬件上验证。

验收：

- SFT/DPO/GRPO都能通过配置或`--deepspeed`选择profile；
- `--deepspeed none`不导入DeepSpeed即可dry-run；
- 声明world size与启动环境不一致时训练前失败；
- stage 2参数offload、同时启用bf16/fp16等非法组合训练前失败；
- 单GPU一步smoke生成真实adapter；
- 三种profile的显存与吞吐只填写实测值。

## 四、AutoDL执行顺序

### 0. 上传仓库和数据

建议仓库路径：

```text
/root/autodl-tmp/audio-codec-llm
```

数据清单可放在：

```text
/root/autodl-tmp/datasets/audio_manifest.v2.jsonl
```

### 1. CPU smoke

```bash
cd /root/autodl-tmp/audio-codec-llm
export PORTFOLIO_V2_MODE=smoke
bash scripts/autodl_v2_bootstrap.sh
bash scripts/autodl_v2_run.sh
```

门：必须出现`RUN_OK mode=smoke`。

### 2. 全量预检

```bash
export PORTFOLIO_V2_MODE=full
export AUDIO_MANIFEST=/root/autodl-tmp/datasets/audio_manifest.v2.jsonl
bash scripts/autodl_v2_bootstrap.sh
bash scripts/autodl_v2_preflight.sh
```

门：必须出现`PREFLIGHT_OK`。

### 3. GRPO短跑策略

第一次建议临时复制配置，将GRPO `max_steps`改为20。20步完成且峰值显存、奖励方差正常
后，再恢复正式配置的300步。不要用没有跑通的长实验消耗租赁时间。

### 4. 完整运行

```bash
nohup bash scripts/autodl_v2_run.sh \
  > outputs/v2/autodl_full.log 2>&1 &
tail -f outputs/v2/autodl_full.log
```

中断恢复：

```bash
bash scripts/autodl_v2_run.sh
```

默认`--resume auto`，无需手填checkpoint。

## 五、GPU实验记录表

开机后将以下真实值填入`run_manifest.md`：

| 项目 | SFT | DPO | GRPO |
| --- | ---: | ---: | ---: |
| 训练样本数 | 待运行 | 待运行 | 待运行 |
| 训练步数 | 待运行 | 待运行 | 待运行 |
| 峰值显存 | 待运行 | 待运行 | 待运行 |
| 耗时 | 待运行 | 待运行 | 待运行 |
| eval loss/reward | 待运行 | 待运行 | 待运行 |
| checkpoint路径 | 待运行 | 待运行 | 待运行 |

DeepSpeed比较使用`docs/deepspeed_comparison_template.csv`，至少比较无DeepSpeed、
ZeRO-2、ZeRO-3 CPU offload；每组保持模型、seed、数据、batch和步数一致。

## 六、最终发布门

以下全部成立才能写入简历：

- 真实checkpoint或LoRA adapter可下载；
- 训练命令、配置、seed和依赖版本公开；
- 有真实模型预测，不只reference自检；
- 有SFT/DPO/GRPO对比；
- 有奖励消融和失败案例；
- README数字与JSON报告一致；
- Hugging Face model card说明基座许可证、训练数据来源和限制；
- 不声称未测的PESQ/STOI/DNSMOS提升。
