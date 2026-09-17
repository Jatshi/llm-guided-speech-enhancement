---
library_name: pytorch
pipeline_tag: audio-to-audio
license: mit
language:
  - en
tags:
  - speech-enhancement
  - mm-dit
  - rectified-flow
  - wavlm
  - structured-conditioning
  - selective-prediction
  - safety-gating
  - stepaudio3-inspired
model-index:
  - name: LSE Prescription-MM-DiT v6 StepAudio3
    results:
      - task:
          type: audio-to-audio
          name: Speech Enhancement
        dataset:
          name: LibriSpeech dev-clean derived paired benchmark
          type: librispeech_asr
        metrics:
          - type: si-sdr-improvement
            name: Oracle minus no-prescription paired SI-SDR improvement
            value: 0.258
          - type: si-sdr-improvement
            name: Predicted minus no-prescription paired SI-SDR improvement
            value: 0.216
---

# LSE Prescription-MM-DiT v6 StepAudio3

This is the measured v6.0 research checkpoint for
[`Jatshi/llm-guided-speech-enhancement`](https://github.com/Jatshi/llm-guided-speech-enhancement).
It is a 51.7M-parameter prescription-conditioned MM-DiT trained as an observed-source rectified
flow for speech enhancement. The v6 training recipe adds strict zero-velocity initialization,
multi-resolution STFT loss, frozen-WavLM semantic consistency, a three-stage curriculum, temporal
`EnhanceScript` conditioning, and explicit raw-versus-safety-gated evaluation.

This package is research evidence, not an SOTA claim. The strongest result is that correct
structured conditions causally improve the model over no prescription and shuffled controls on
genuinely corrupted audio. DeepFilterNet3 remains stronger on several perceptual metrics.

## Package layout

| Path | Purpose |
| --- | --- |
| `mmdit/checkpoint-best.pt` | Step-11,750 model, optimizer lineage, model/flow/codec config and curriculum metadata. |
| `configs/mmdit_v6_stepaudio3_32gb.json` | Exact 12,000-step training and six-arm evaluation config. |
| `evaluation/evaluation_report.json` | Full aggregate, median, corrupted-only and per-degradation summaries. |
| `evaluation/diagnostic_summary.json` | Paired bootstrap comparisons and v5-to-v6 analysis. |
| `evaluation/per_sample.csv` | 1,200 arm-level rows for 200 held-out examples. |
| `evaluation/summary.csv` | Compact aggregate table. |
| `training/training_report.json` | Formal training environment and completion record. |
| `training/semantic_teacher_probe.json` | Frozen-teacher gradient-connectivity evidence. |
| `SHA256SUMS` | Integrity hashes for every packaged file. |

PyTorch checkpoints can execute code during deserialization. Download only from a trusted source
and verify `SHA256SUMS` before loading.

## Architecture and training

- Input/output representation: two-channel real/imaginary complex STFT, 16 kHz, 2-second crops.
- Flow path: `x_t = (1-t) * noisy_latent + t * clean_latent`.
- Network: 8 MM-DiT blocks, width 384, 8 heads, 51,717,122 trainable parameters.
- Structured condition: up to 16 prescription / temporal `EnhanceScript` tokens.
- Identity start: AdaLN modulation and final velocity head are initialized to exact zero.
- Auxiliary objectives: 256/512/1024-point MR-STFT and frozen WavLM layer-9 cosine consistency.
- Curriculum: oracle alignment (steps 1-4,000), safety alignment (4,001-9,000), joint cooldown
  (9,001-12,000).
- Hardware: one NVIDIA RTX 4080 SUPER 32 GB; bfloat16; batch size 12.
- Data: 2,000 LibriSpeech dev-clean derived pairs with 1,600/200/200 speaker-disjoint split.

The WavLM teacher is not redistributed. It remains frozen and is needed only to reproduce semantic
training, not for MM-DiT inference.

## Held-out results

The test set contains 40 examples each of clean, white noise, pink noise, reverb, and telephone
band-limiting. Clean identity pairs have approximately 143 dB input SI-SDR, so tiny waveform changes
produce extreme negative SI-SDR improvement. Enhancement quality is therefore reported on the 160
genuinely corrupted examples, while clean remains a separate safety slice.

### Raw model output on corrupted examples

| Arm | SI-SDR improvement | STOI | PESQ | Evaluation fallback |
| --- | ---: | ---: | ---: | ---: |
| DeepFilterNet3 | +0.374 dB | 0.8719 | 1.9886 | 0% |
| MM-DiT no prescription | +0.005 dB | 0.8578 | 1.3683 | 0% |
| MM-DiT oracle prescription | **+0.263 dB** | 0.8647 | 1.3149 | 0% |
| MM-DiT predicted prescription | **+0.221 dB** | 0.8578 | 1.3684 | 70.6% |
| MM-DiT shuffled prescription | -0.114 dB | 0.8575 | 1.3599 | 0% |
| Spectral subtraction | -1.124 dB | 0.8408 | 1.3459 | 0% |

The fallback column belongs to the separate safety-gated output. Values in the SI-SDR column above
are always computed on raw candidates so that fallback cannot hide model failures.

### Paired causal comparisons

20,000 paired bootstrap resamples over the same 160 corrupted examples:

| Comparison | Mean difference | Win rate | 95% confidence interval |
| --- | ---: | ---: | --- |
| Oracle minus no prescription | +0.258 dB | 64.4% | [+0.089, +0.424] |
| Oracle minus shuffled | +0.376 dB | 75.6% | [+0.195, +0.552] |
| Predicted minus no prescription | +0.216 dB | 68.1% | [+0.142, +0.294] |

These comparisons support the claim that the network uses the structured prescription. They do not
support universal enhancement superiority.

Matching source release: [`v6.0.0`](https://github.com/Jatshi/llm-guided-speech-enhancement/releases/tag/v6.0.0).

## Inference

Clone the matching source branch/tag, place the checkpoint under `mmdit/`, and run:

```bash
python -m lse_v2.mmdit.infer \
  --checkpoint mmdit/checkpoint-best.pt \
  --input noisy.wav \
  --prescription examples/mmdit_prescription.json \
  --output enhanced.wav
```

The checkpoint embeds model, flow, codec and curriculum metadata. The frozen WavLM teacher and
optimizer state are not used during inference.

## Limitations

- Absolute SI-SDR gains are small, and PESQ/STOI do not surpass DeepFilterNet3.
- The acoustic router diagnoses 158/200 test items correctly, but the current predicted action is
  always `spectral_subtraction`; dereverb and bandwidth-extension actions are not yet implemented
  end to end.
- Predicted-prescription fallback is 70.6% on corrupted samples and 76.5% on the full set. The
  current confidence threshold discards some candidates with small positive gains.
- The benchmark is small, synthetic, English-only, and based on two-second crops.
- Reference-based rollback is an evaluation mechanism. Production deployment requires a calibrated
  no-reference quality and safety estimator.
- The time-local `EnhanceScript` contract is implemented, but this run uses bootstrapped full-span
  segments rather than a dedicated frame-local corruption dataset.
- Independent identity/MR-STFT/semantic training ablations and executed multi-candidate Planner
  preference learning remain future work.

## Integrity

```text
720eecf88cdc4b878b640144fcbbc8f49361bf7920577bc03a8e3da3dc98c9dd  mmdit/checkpoint-best.pt
3f57e27aa2e2fbda23a37ded7965f024f6fc30053d8aab2938218487b173d9ee  evaluation/evaluation_report.json
488e53f6246832c4b3ef0847192962d9bd2d4d80c845fdf4c0a45dfbe7046336  evaluation/diagnostic_summary.json
c77b9500c9dedd9eb7f34852a48ee34579ad33a7f05ec47e32bbc1875164ff48  training/training_report.json
```

See the repository execution report for the complete timeline, two recovered engineering bugs, and
the interpretation boundary between raw model quality and safety-gated output.
