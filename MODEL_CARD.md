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
  - lora
  - speech-enhancement
  - audio
  - safety-gating
  - deepspeed
---

# LLM-Guided Speech Enhancement 4.0 — Native-Audio SFT

This release contains the evidence-selected SFT LoRA adapter and audio projector from the 4.0
native-audio training run. It pairs a frozen `openai/whisper-small` encoder with
`Qwen/Qwen2.5-1.5B-Instruct`: Whisper hidden states are compressed into 16 continuous prefix tokens,
and the language model emits a validated JSON DSP prescription.

The language model does not directly synthesize a clean waveform. A deterministic DSP executor
validates and applies its prescription, computes objective metrics against the available clean
reference, and returns the original input whenever the candidate fails the safety gate.

Source and reproducibility code:
[Jatshi/llm-guided-speech-enhancement](https://github.com/Jatshi/llm-guided-speech-enhancement).

## Why the SFT checkpoint is released

All three training stages completed and their artifacts remain auditable. The final checkpoint was
not selected merely because it was last:

| Stage | Training status | Held-out structural result | Release decision |
|---|---|---:|---|
| SFT | 1,500 steps completed | 1,416/1,416 valid JSON | **selected** |
| conservative DPO | 800 steps completed | canary output contains invalid numeric placeholders | rejected |
| GRPO | 300 steps completed | 0/1,416 valid JSON; mean reward 0 | rejected |

GRPO reported 1,200 saturated groups and zero mean reward. The 1,416 GRPO failures are preserved as
negative evidence; this model card does not claim preference-optimization or RL improvement.

## Training data

- Clean speech: LibriSpeech `train-clean-100` through a documented ModelScope mirror.
- Environmental noise: ESC-50.
- Physical materialization: 20,000 noisy WAV records, all with clean reference, source provenance,
  deterministic materialization seed, auxiliary acoustic measurements, SFT target, and synthetic
  safety-negative preference response.
- Split: 16,164 train, 2,420 evaluation, and 1,416 test examples.
- Licenses: LibriSpeech is CC BY 4.0; ESC-50 is CC BY-NC 3.0. The materialized dataset is therefore
  not bundled in this model repository and must be reconstructed from the documented sources.
- Preference labels are rule-generated safety negatives, not human preference judgments.

## Optimization

Training ran on one NVIDIA RTX 4080 SUPER with 32,760 MiB VRAM using QLoRA/NF4, BF16 compute,
gradient accumulation, and DeepSpeed ZeRO-2 at `world_size=1`.

| Stage | Steps | Initial loss | Final loss | Runtime |
|---|---:|---:|---:|---:|
| SFT | 1,500 | 2.314612 | 0.087327 | 3,017.39 s |
| DPO | 800 | 0.406933 | 0.325083 | 3,525.89 s |
| GRPO | 300 | 1.3355e-5 | 4.6299e-5 | 9,689.05 s |

Single-GPU ZeRO-2 is integration and optimizer-state evidence, not multi-GPU scaling evidence. A
ZeRO-3 CPU-offload configuration is included for memory experiments but is not described as single-
GPU parallelism.

## Held-out evaluation

The release evaluation covers all 1,416 test records. Cached Whisper embeddings were reused and
generation was performed in deterministic batches of 16.

| Metric | Result |
|---|---:|
| Valid JSON | 1,416/1,416 |
| Failed predictions | 0 |
| Inference wall time | 843.95 s |
| Throughput | 1.6778 samples/s |
| Amortized mean latency | 595.97 ms/sample |
| p50 / p95 latency | 586.20 / 667.53 ms/sample |

The executable DSP benchmark reports candidate deltas before rollback:

| Objective result | Mean delta |
|---|---:|
| SI-SDR | **+0.4509 dB** |
| PESQ | **+0.04068** |
| STOI | **+0.002198** |
| Safety accept rate | **66.74%** |
| Safety rollback rate | **33.26%** |

All four SNR buckets (`<5`, `5–10`, `10–20`, and `>=20` dB) have positive mean deltas for all three
available metrics. DNSMOS, WER, and speaker similarity are not reported because their backends were
not configured; missing metrics are not replaced with fabricated values.

## Required components

This repository contains:

- the SFT PEFT adapter;
- `audio_projector.pt`, which maps Whisper hidden states into the 16-token language prefix;
- model-selection, prediction, benchmark, and run manifests.

It does not redistribute Qwen2.5-1.5B-Instruct or Whisper-small. Load those base models from their
original repositories and use `lse_v2.native_inference.NativeAudioPlanner` from the source repo.

## Safety behavior

1. The generated text must parse as one JSON object.
2. The prescription must contain `diagnosis`, `actions`, `rationale`, and `confidence`.
3. Every DSP action and parameter must pass a strict allowlist and numeric bounds.
4. The proposed waveform is measured against the noisy input and clean reference in this benchmark.
5. A candidate with negative SI-SDR gain is rolled back to the original waveform.

The 33.26% rollback rate is an observed safety behavior, not a hidden failure count.

## Known limitations

- Evaluation uses English LibriSpeech speech mixed with ESC-50 noise; other languages, microphones,
  rooms, codecs, and conversational overlap remain out of distribution.
- Mean gains are positive but modest. This is a control-policy project, not a state-of-the-art neural
  waveform enhancer.
- Clean references are available in the benchmark. A production deployment must replace reference-
  based gating with calibrated no-reference metrics or human review.
- DPO and GRPO regressed output syntax in this run. They are research artifacts, not release models.
- Rule-generated preferences are too regular to establish human-alignment quality.
- The original STFT/ISTFT implementation used inconsistent overlaps; the repaired version is guarded
  by a waveform round-trip regression test. Metrics in this card use only the repaired evaluation.

## Reproducibility

Use `configs/native_audio_autodl_32gb.json` and follow the
[4.0 AutoDL runbook](https://github.com/Jatshi/llm-guided-speech-enhancement/blob/main/docs/AUTODL_V4_RUNBOOK_ZH.md).
The release bundle includes machine-readable reports and SHA-256 checksums. Exact package versions
are recorded in `environment.freeze.txt`.

The source code is MIT licensed. Use of the adapter and reconstructed data remains subject to the
Qwen, Whisper, LibriSpeech, and ESC-50 terms.
