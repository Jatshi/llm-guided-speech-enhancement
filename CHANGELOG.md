# Changelog

## 4.1.0 - 2026-09-09

- Recover GRPO from the verified v4.0 SFT adapter instead of the malformed DPO adapter.
- Add bounded partial-JSON shaping without rewarding question-mark placeholders.
- Add an SFT anchor loss to preserve executable JSON during policy optimization.
- Add generation canaries, reward-saturation diagnostics, fail-fast collapse reports, and
  held-out SFT-vs-GRPO promotion gates.
- Add a recovery-only runner that reuses cached Whisper embeddings and skips completed SFT/DPO.
- Add continuous parameter calibration against the deterministic synthetic-degradation target so
  valid but suboptimal prescriptions no longer receive identical reward.
- Complete the 300-step 12GB-GPU run: 99.15% sampled JSON validity, 99.83% non-saturated groups,
  and no collapse report.
- Pass the 1,416-example promotion gate with 100% valid JSON, zero placeholders, and mean reward
  improving from 0.95382 (SFT) to 0.97632 (recovered GRPO).

## 4.0.0 - 2026-08-26

- Completed the 20,000-waveform native-audio run and objective DSP generalization evaluation.
- Published the held-out-selected SFT adapter and retained DPO/GRPO as auditable negative results.

## 3.0.0 - 2026-08-13

- Added Whisper hidden-state pooling and a trainable native-audio prefix projector.
- Added bounded DSP execution and execute–measure–revise/rollback control.
- Added explicit unavailable metric semantics and DeepSpeed single-GPU contracts.
- Published the CUDA-validated 7.1MB native-audio projector on Hugging Face.
- Added separate 3.0 release notes and a deep Chinese learning, failure, and interview guide.

## 2.0.0 - 2026-07-28

- Added versioned SFT, DPO, and GRPO data/reward pipelines.
- Added conservative, auditable speech-enhancement prescriptions and stage evaluation.
- Published the Qwen2.5-1.5B GRPO LoRA and complete from-scratch Chinese guide.
