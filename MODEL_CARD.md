---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: mit
tags:
  - speech-enhancement
  - audio
  - lora
  - dpo
---

# LLM-Guided Speech Enhancement adapters

This repository hosts the final LoRA adapter for **LLM-Guided Speech Enhancement**. The `dpo-adapter/` checkpoint is based on `Qwen/Qwen2.5-7B-Instruct`.

## Intended use

Given structured acoustic evidence, the adapter produces an interpretable response comprising a degradation diagnosis, a DSP-oriented enhancement strategy, and a rationale. It is intended as a research and prototyping control layer, not as a standalone waveform denoiser.

## Training

Training examples were generated from AISHELL-1 clean-speech metadata and programmatically sampled degradations. Supervised fine-tuning was followed by Direct Preference Optimization. The synthetic construction makes the response format highly regular.

## Limitations

Reported held-out format and diagnosis metrics are in-distribution metrics against programmatically constructed targets. They do not demonstrate generalization to arbitrary recordings or superiority on perceptual speech-enhancement benchmarks. Use the adapters with the Qwen and AISHELL-1 licenses/terms in mind.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "jatshi/llm-guided-speech-enhancement", subfolder="dpo-adapter")
```

Source code: https://github.com/Jatshi/llm-guided-speech-enhancement
