---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: mit
language:
  - en
  - zh
tags:
  - qwen2.5
  - lora
  - speech-enhancement
  - audio
  - dpo
  - grpo
  - deepspeed
---

# Audio-Codec-LLM 2.0 — Qwen2.5-1.5B GRPO LoRA

This repository contains the final LoRA adapter from the Audio-Codec-LLM 2.0
training chain:

```text
Qwen2.5-1.5B-Instruct
  → supervised fine-tuning
  → conservative DPO
  → verifiable GRPO
```

The model reads structured acoustic evidence and produces an auditable JSON
enhancement prescription containing a degradation diagnosis, DSP actions,
rationale, and confidence. It is a language-model control layer for speech
enhancement research. It does **not** directly ingest or restore raw waveforms.

Source code and reproducibility scripts:
[Jatshi/llm-guided-speech-enhancement](https://github.com/Jatshi/llm-guided-speech-enhancement).

## Training data

- Source: AISHELL-1 clean speech, with source URL, license, checksum, and seed
  recorded in the dataset provenance.
- 40,000 clean source files were sampled with seed 42.
- Three deterministic degradation configurations were generated per source,
  yielding 120,000 versioned acoustic-evidence records.
- Each SFT, DPO, and GRPO contract contains 114,000 training records and 6,000
  held-out records.
- 200 noisy waveforms were materialized for reproducibility checks. The other
  rows are explicitly marked as clean-audio proxies paired with reproducible
  synthetic-degradation metadata; they are not represented as physically
  materialized noisy recordings.
- DPO rejected responses are programmatically generated safety negatives. They
  are useful for controlled alignment experiments but are not a substitute for
  human preference labels.

## Optimization

All stages ran on one NVIDIA RTX 4090 24 GiB in BF16 with LoRA rank 16 and
DeepSpeed ZeRO-2 compatibility enabled at `world_size=1`.

| Stage | Data/steps | Micro batch × accumulation | Objective |
| --- | ---: | ---: | --- |
| SFT | 114,000 rows, 1 epoch | 8 × 2 | token cross-entropy |
| cDPO | 114,000 pairs, 1 epoch | 8 × 2 | `beta=0.01`, label smoothing `0.1` |
| GRPO | 300 optimizer steps | 4 × 4, 2 generations | five deterministic reward components |

The initial DPO trial with `beta=0.1` was saturated at step zero. Lowering
`beta` to 0.01 restored a useful gradient, but standard DPO then drove rejected
log-probabilities toward negative infinity. The final run therefore uses
conservative DPO with label smoothing 0.1. Its finite optimum is
`log((1-0.1)/0.1) ≈ 2.20`, preventing unbounded preference margins while
retaining the full dataset and epoch.

## Verified results

The completed SFT stage processed all 114,000 rows in 5,243.8 seconds:

| Metric | Value |
| --- | ---: |
| train loss | 0.09630 |
| held-out loss | 0.08168 |
| held-out token accuracy | 0.96517 |
| training throughput | 21.74 samples/s |

<!-- FINAL_EVAL_TABLE: replaced after cDPO, GRPO, and the four-stage 200-row holdout matrix finish. -->

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "jatshi/Audio-Codec-LLM-Qwen2.5-1.5B-GRPO-LoRA"

tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
```

Use the system prompt and JSON contract in the source repository. The adapter
expects text-form acoustic evidence; passing an audio file path alone is not
sufficient.

## Evaluation boundary

- The final comparison uses the same 200 uniformly sampled held-out IDs for the
  base, SFT, DPO, and GRPO checkpoints and records the selection SHA-256.
- These are in-distribution structured-prescription metrics, not PESQ, STOI,
  DNSMOS, or a listening test.
- The 6,000-row split is a held-out split from the programmatically constructed
  dataset. It is not an independently collected real-world test corpus.
- Single-GPU ZeRO results prove integration and memory/throughput behavior only.
  They do not establish multi-GPU scaling.

## Limitations and safety

- Prescriptions should be validated before controlling a production audio
  pipeline; unsafe parameter combinations can damage intelligibility.
- Acoustic labels for proxy rows come from degradation configuration, not
  direct feature extraction from a materialized noisy waveform.
- Synthetic preference pairs are regular and easier than human disagreements.
- Generalization to unseen languages, microphones, rooms, and degradation
  mixtures has not been established.
- The model can emit valid JSON that is acoustically inappropriate. Downstream
  parameter bounds and abstention policies remain necessary.

The source code is MIT licensed. The adapter remains subject to the base model
and AISHELL-1 terms in addition to this repository's license.
