---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: audio-text-to-text
license: mit
language:
  - en
tags:
  - qwen2.5
  - whisper
  - qlora
  - grpo
  - speech-enhancement
  - safety-gating
  - deepspeed
---

# LLM-Guided Speech Enhancement 4.1 — Recovered GRPO

This repository contains the promotion-gate-selected GRPO LoRA adapter and audio projector from
LLM-Guided Speech Enhancement 4.1. A frozen Whisper-small encoder is compressed into 16 continuous
prefix tokens for Qwen2.5-1.5B-Instruct. The language model emits a strict JSON DSP prescription;
it does not directly synthesize waveform samples.

Source, exact configuration, tests, and the Chinese failure analysis:
[Jatshi/llm-guided-speech-enhancement](https://github.com/Jatshi/llm-guided-speech-enhancement).

## Why this GRPO model is published

The v4.0 GRPO run collapsed because every candidate in every group received zero reward. Version
4.1 starts from the verified SFT adapter, keeps an SFT anchor, rejects saturated pre-training
canaries, and continuously calibrates otherwise-valid action parameters against the deterministic
synthetic-degradation target. The target is data-derived; no LLM judge is used.

| Promotion evidence | Result |
|---|---:|
| Pre-training canary | 48/48 valid JSON; 12/12 non-saturated groups |
| Training | 300 optimizer steps; 1,200 groups |
| Sampled valid JSON | 99.1458% |
| Non-saturated groups | 99.8333% |
| Held-out predictions | 1,416/1,416 successful |
| Placeholder rate | 0% |
| SFT mean verifiable reward | 0.953821 |
| GRPO mean verifiable reward | **0.976320** |
| Promotion gate | **passed** |

## Objective DSP benchmark

All 1,416 held-out predictions were executed by the bounded DSP engine. Candidates that failed the
SI-SDR safety condition were rolled back to the original noisy waveform.

| Metric | Result |
|---|---:|
| Safety accept / rollback | 68.15% / 31.85% |
| Mean SI-SDR gain | **+0.35468 dB** |
| Mean PESQ gain | **+0.04056** |
| Mean STOI gain | **+0.00260** |
| Mean / P95 prescription latency | 786.15 / 873.03 ms |

All four SNR buckets (`<5`, `5–10`, `10–20`, and `>=20` dB) have positive mean gains for all three
metrics. These are objective, reference-based benchmark measurements rather than listening-test
claims.

## Files

- `adapter/`: PEFT LoRA adapter and tokenizer files;
- `audio_projector.pt`: Whisper-hidden-state to 16-token prefix projection;
- `reports/`: canary, training, prediction, generalization, acceptance, and environment evidence;
- `README.md`: this model card.

The base Qwen and Whisper weights are not redistributed.

## Reproducibility

Use `configs/native_audio_autodl_grpo_recovery_12gb.json` from the source repository. The verified
runtime used Python 3.10.8, PyTorch 2.5.1, Transformers 4.48.3, PEFT 0.14.0, Accelerate 1.2.1, and
DeepSpeed 0.16.3 on one RTX 3080 Ti 12GB. DeepSpeed ZeRO-2 at `world_size=1` is runtime integration
evidence, not a multi-GPU scaling claim.

## Limitations

- Evaluation is English LibriSpeech speech with physically materialized synthetic degradations;
  real rooms, microphones, overlap, and other languages need separate testing.
- Positive objective gains are modest and do not establish state-of-the-art enhancement quality.
- The benchmark safety gate has clean references. Production use needs a calibrated no-reference
  gate or human review.
- Preferences and calibration targets are programmatically generated, not human preference labels.

The source code is MIT licensed. Use of the adapter and reconstructed data remains subject to the
Qwen, Whisper, LibriSpeech, and ESC-50 terms.
