"""Generate deterministic offline predictions from a trained LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, set_global_seed
from .contracts import GRPO_SCHEMA, validate_alignment_record
from .io import read_jsonl, write_jsonl


def generate_predictions(
    config_path: str | Path,
    dataset_path: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
    *,
    max_samples: int | None = None,
) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    set_global_seed(int(config["project"]["seed"]))
    rows = read_jsonl(dataset_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    for row in rows:
        validate_alignment_record(row)
        if row["schema_version"] != GRPO_SCHEMA:
            raise ValueError("Prediction input must use lse.grpo.v2")
    model_id = config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = str(config["model"].get("dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        torch_dtype=dtype,
        device_map=config["model"].get("device_map", "auto"),
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    predictions: list[dict[str, Any]] = []
    max_new_tokens = int(config.get("evaluation", {}).get("max_new_tokens", 256))
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        predictions.append({"sample_id": row["sample_id"], "response": response})
    write_jsonl(output_path, predictions)
    return len(predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl_4090.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = generate_predictions(
        args.config,
        args.dataset,
        args.adapter,
        args.output,
        max_samples=args.max_samples,
    )
    print(json.dumps({"predictions": count, "output": str(Path(args.output).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
