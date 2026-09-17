---
base_model:
  - Qwen/Qwen2.5-1.5B-Instruct
library_name: pytorch
pipeline_tag: audio-to-audio
license: mit
language:
  - en
tags:
  - speech-enhancement
  - mm-dit
  - rectified-flow
  - peft
  - selective-prediction
  - safety-gating
---

# LSE Prescription-MM-DiT v5

This release is a three-component, safety-gated speech-enhancement research system:

1. a 51.7M-parameter Prescription-MM-DiT trained as an observed-source residual rectified flow;
2. a Qwen2.5-1.5B LoRA planner that emits strict JSON actions and explanations;
3. an ExtraTrees acoustic router calibrated on the validation split and restricted to executable
   white/pink-noise actions.

The matching source is available at
[`Jatshi/llm-guided-speech-enhancement`](https://github.com/Jatshi/llm-guided-speech-enhancement),
tag `v5.0.0`.

## Package layout

| Path | Purpose |
|---|---|
| `mmdit/checkpoint-best.pt` | Step-10,000 MM-DiT checkpoint and embedded architecture/flow config. |
| `planner-adapter/` | Full-sequence SFT LoRA and tokenizer files for Qwen2.5-1.5B-Instruct. |
| `router/extra_trees_safe.joblib` | Train-only fitted acoustic classifier used by the hybrid planner. |
| `evaluation/` | Final report, summary, per-sample table, and training/stage manifests. |
| `configs/` | Exact final training and safety-gated evaluation configurations. |

Only load the PyTorch checkpoint and joblib router from a trusted source: both serialization formats
can execute code during deserialization. Verify hashes before use.

## Training and data

- Public source: LibriSpeech dev-clean.
- Conditions: clean, white noise, pink noise, reverberation, and telephone-band filtering.
- Split: 1,600 train / 200 validation / 200 test, with 25 / 4 / 4 disjoint speakers.
- Test labels were not used to fit MM-DiT, LoRA, or ExtraTrees.
- MM-DiT: 10,000 steps, best validation loss 0.0735645, 1,803.97 seconds on one RTX 4080 SUPER
  32 GB.
- Planner SFT: 300 steps; 200/200 test generations were strict-valid JSON.
- Confidence threshold: 0.70, selected on validation only.

## Held-out evaluation

The final test ran 200 examples across six arms (1,200 rows). The safety-gated predicted arm reports:

| Metric | Result |
|---|---:|
| SI-SDR improvement | +0.1139 dB |
| SNR improvement | +0.1319 dB |
| PESQ | 2.0268 |
| STOI | 0.8812 |
| Mean MM-DiT latency | 71.97 ms/sample |
| Fallback rate | 76.5% |

DeepFilterNet3 achieved +0.374 dB mean SI-SDR improvement on the non-clean subset, compared with
+0.142 dB for this system. The v5 claim is therefore **safe selective enhancement**, not universal
or state-of-the-art denoising. Aggregate DeepFilterNet SI-SDRi is distorted by the clean condition's
near-degenerate reference SI-SDR and should not be used as the sole comparison.

## Inference

Download the model package into the source repository and run:

```bash
python -m lse_v2.mmdit.infer \
  --checkpoint mmdit/checkpoint-best.pt \
  --input noisy.wav \
  --prescription examples/mmdit_prescription.json \
  --output enhanced.wav
```

To reproduce the final hybrid planner and full six-arm benchmark, follow
[`docs/MMDIT_AUTODL_RUNBOOK.md`](https://github.com/Jatshi/llm-guided-speech-enhancement/blob/v5.0.0/docs/MMDIT_AUTODL_RUNBOOK.md).

## Limitations

- The trained enhancement action currently supports white/pink noise; clean, reverb, telephone, and
  low-confidence cases are intentionally rejected or returned unchanged.
- The pure LLM diagnosis accuracy is 29.5%; the published 79% value belongs to the ExtraTrees
  acoustic router, not the LLM.
- The test set is small and English-only. No claim is made for multilingual, overlapping-speaker,
  far-field, or real production traffic.
- The benchmark's reference-based rollback is an evaluation tool. Production use requires a
  calibrated no-reference quality/safety estimator.
- The package does not redistribute Qwen base weights or LibriSpeech audio.

## Integrity

Key SHA-256 values:

```text
d8e1677d15b3ad6b53b87a1a08c3b95c4dc1c028dfa237267646d5d7b8306c21  evaluation_report.json
777a4ca76efe2d00ad66a74e5cf198c018f7639951b93c04630a6c6bed51b27f  extra_trees_safe.joblib
868bf0918a649f3e7b0feb7638dfface83dd2abcbd5274510823db4cb43832e7  planner adapter_model.safetensors
```

See the packaged `SHA256SUMS` for the MM-DiT checkpoint and every uploaded release file.
