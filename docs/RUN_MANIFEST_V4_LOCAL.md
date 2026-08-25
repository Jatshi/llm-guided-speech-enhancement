# 4.0 Local Implementation Run Manifest

- Date: 2026-08-24 (Asia/Shanghai)
- Upstream: `a2a4ca57624a957445dbf4d475c116d57e638800`
- Working branch: `codex/production-hardening`
- Scope: LLM speech-enhancement production hardening only
- Source plan: `两个项目夯实规划_LLM语音增强与EvidenceAgent.md`
- Baseline before modifications: 51 tests passed
- Final local tests: 87 passed
- Static gates: Ruff check/format and `git diff --check` passed
- Packaging: `lse_v2-4.0.0-py3-none-any.whl` built successfully; SHA-256
  `1d4de5c5b12292d92ef861479ee291a7c7b123c5e7cf0780fab0e4cbfa47e273`

## Implemented

Production DSP, waveform output/rollback, global and frame metrics, physical materialization,
source-level split, native Whisper-prefix data/training/inference, SFT/cDPO/GRPO, resumable
DeepSpeed-compatible stages, generalization slices, windowed streaming, and authenticated serving.

## Evidence boundary

- 4.0 GPU training: **not run**.
- 4.0 full-chain quality metrics: **not available**.
- 4.0 AutoDL artifacts/model weights: **not created**.

The 2.0/3.0 GPU results in the historical root `run_manifest.md` do not transfer to the new 4.0
native-audio chain. Single-GPU ZeRO is runtime/offload compatibility, not multi-GPU evidence.
