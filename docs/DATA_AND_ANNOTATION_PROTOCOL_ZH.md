# 数据、偏好标注与人工听测协议

## 数据许可与来源

每个 catalog 行必须有 `dataset`、`license`、绝对 `audio_path`、source/speaker ID 和文件
SHA-256。公开发布前逐项复查上游条款；代码不会自动下载 DNS、VoiceBank、DEMAND、WHAM!
或 Clarity，防止把“可研究使用”误当“可任意再分发”。

## 切分

先按 speaker/source 哈希切分，再做退化变体。同一 clean source 的所有噪声、RIR、SNR
版本只能在一个 split。test source 不参与 prompt 模板、阈值选择或 reward 权重调节。

## cDPO 人工偏好 JSONL

```json
{"sample_id":"...","chosen_text":"{...}","rejected_text":"{...}","annotator_id":"A03","rationale":"chosen removes HVAC line while preserving speech; rejected over-suppresses 2–4 kHz"}
```

chosen 必须通过 DSP 安全解析器；rejected 必须是 JSON 对象，但可以包含错误诊断、过处理或
不合理动作，供偏好训练学习。annotator 不应看到系统随机种子或隐藏真值，只听音频、看
可测特征并比较候选。至少抽取一部分样本双人标注，报告一致率和分歧仲裁数量。

## 盲听表

随机打乱 noisy、candidate A/B，不显示模型阶段。每个样本按 1–5 分记录：噪声残留、语音
自然度、可懂度、音乐噪声、整体偏好，并提供“二者无差异”。报告样本数、听者数、置信
区间和显著性；不能只展示最好听的三个例子。

```bash
python -m lse_v2.listening_test build \
  --sample-results outputs/native_v4/generalization/sample_results.jsonl \
  --output-dir outputs/native_v4/listening

python -m lse_v2.listening_test aggregate \
  --responses outputs/native_v4/listening/responses.jsonl \
  --blind-key outputs/native_v4/listening/blind_key.json \
  --output outputs/native_v4/listening/report.json
```

试听进行期间把 `blind_key.json` 与试听者隔离，全部标注冻结后再揭盲。
