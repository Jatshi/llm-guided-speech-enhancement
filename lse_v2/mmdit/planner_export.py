"""Run the existing LoRA planner without oracle labels and export JSON prescriptions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from lse_v2.contracts import SYSTEM_PROMPT, extract_audio_features
from lse_v2.io import read_jsonl, write_jsonl
from lse_v2.rewards import parse_prescription

from .contracts import validate_prescription


def _close_truncated_json(text: str) -> str | None:
    """Close only unmatched JSON containers at end-of-generation.

    This deliberately does not add fields, move values, quote strings, or change
    any model-produced semantics. It is therefore suitable as an auditable
    transport repair for otherwise complete generations cut off before their
    final ``]``/``}`` tokens.
    """

    candidate = text.strip()
    if not candidate.startswith("{"):
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for character in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            stack.append("}")
        elif character == "[":
            stack.append("]")
        elif character in "}]":
            if not stack or stack[-1] != character:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    return candidate + "".join(reversed(stack))


def parse_planner_prediction(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse a planner response and report strict/repaired/invalid provenance."""

    parsed = parse_prescription(text)
    status = "strict"
    if parsed is None:
        repaired = _close_truncated_json(text)
        parsed = parse_prescription(repaired) if repaired is not None else None
        status = "repaired" if parsed is not None else "invalid"
    if parsed is None:
        return None, "invalid"
    try:
        validate_prescription(parsed, "planner_prediction")
    except Exception:
        return None, "invalid"
    return parsed, status


def build_planner_prompt(audio_path: str | Path, sample_rate: int = 16000) -> str:
    features = extract_audio_features(audio_path, sample_rate)
    evidence = {
        "sample_rate": sample_rate,
        "measured_features": features,
        "privileged_degradation_labels_in_prompt": False,
    }
    return (
        "Listen to the measured evidence and return a safe executable enhancement "
        "prescription. Do not infer a known dataset condition from the sample id.\n"
        f"AUXILIARY_EVIDENCE={json.dumps(evidence, sort_keys=True)}"
    )


def export_predictions(
    pairs: Path,
    output: Path,
    *,
    base_model: str,
    adapter: Path,
    max_new_tokens: int = 256,
    splits: set[str] | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_records = read_jsonl(pairs)
    records = [row for row in all_records if splits is None or row.get("split") in splits]
    if not records:
        requested = "all" if splits is None else ",".join(sorted(splits))
        raise ValueError(f"no records found for requested split(s): {requested}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter).eval()
    rows = []
    invalid = 0
    strict_valid = 0
    repaired_valid = 0
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        rendered = []
        for record in batch:
            prompt = build_planner_prompt(
                record["audio"]["noisy_path"], int(record["audio"]["sample_rate"])
            )
            rendered.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        encoded = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        raw_responses = tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        for local_index, (record, raw) in enumerate(
            zip(batch, raw_responses, strict=True), start=1
        ):
            raw = raw.strip()
            parsed, parse_status = parse_planner_prediction(raw)
            if parse_status == "strict":
                strict_valid += 1
            elif parse_status == "repaired":
                repaired_valid += 1
            else:
                invalid += 1
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "prediction": parsed,
                    "raw_response": raw,
                    "parse_status": parse_status,
                    "latency_ms": elapsed_ms / len(batch),
                    "used_oracle_labels": False,
                }
            )
            index = offset + local_index
            print(
                f"planner prediction {index}/{len(records)} valid={parsed is not None}",
                flush=True,
            )
    write_jsonl(output, rows)
    report = {
        "records": len(rows),
        "valid": len(rows) - invalid,
        "strict_valid": strict_valid,
        "repaired_valid": repaired_valid,
        "invalid": invalid,
        "valid_rate": (len(rows) - invalid) / len(rows) if rows else 0.0,
        "source_records": len(all_records),
        "requested_splits": sorted(splits) if splits is not None else None,
        "batch_size": batch_size,
        "base_model": base_model,
        "adapter": str(adapter.resolve()),
        "oracle_labels_exposed": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        dest="splits",
        help="only predict a selected split; repeat to select multiple splits",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_predictions(
                args.pairs,
                args.output,
                base_model=args.base_model,
                adapter=args.adapter,
                max_new_tokens=args.max_new_tokens,
                splits=set(args.splits) if args.splits else None,
                batch_size=args.batch_size,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
