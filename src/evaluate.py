# -*- coding: utf-8 -*-
"""
评估：加载 base + DPO（或 SFT）LoRA adapter，在带真实退化配置的评估子集上
逐条生成增强策略，并计算：
- 退化诊断准确率：模型是否正确识别噪声类型（及混响/频带限制）；
- 策略格式正确率：能否从输出中解析出可执行 DSP 参数（dB / Q / Hz）；
- 关键词覆盖率：诊断/策略/理由三段结构是否完整。

依赖 metadata.json 中带 audio_path 的评估样本（含 ground-truth degradation_config）。
"""
import os
import re
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degradation import config_to_feature_text, NOISE_ZH

PROJECT = os.environ.get("LSE_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT, "models", "Qwen2.5-7B-Instruct"))
DPO_ADAPTER = os.path.join(PROJECT, "outputs/dpo/final")
SFT_ADAPTER = os.path.join(PROJECT, "outputs/sft/final")
METADATA = os.path.join(PROJECT, "data/training/metadata.json")
RESULT = os.path.join(PROJECT, "outputs/eval_results.json")

SYSTEM_PROMPT = "你是一个专业的语音增强专家，擅长分析音频退化类型、生成可执行的 DSP 增强策略，并解释理由。"

# 噪声类型 -> 判定诊断正确的关键词集合（命中任一即算识别到该类型）
NOISE_KEYWORDS = {
    "white": ["白噪声", "宽带", "全频段"],
    "pink": ["粉红", "低频", "轰鸣"],
    "hvac": ["空调", "窄带", "200-500", "带阻"],
    "cafe": ["咖啡", "背景人声", "chatter", "中高频"],
}


def pick_adapter():
    if os.path.isdir(DPO_ADAPTER) and os.path.exists(os.path.join(DPO_ADAPTER, "adapter_config.json")):
        return DPO_ADAPTER, "DPO"
    if os.path.isdir(SFT_ADAPTER) and os.path.exists(os.path.join(SFT_ADAPTER, "adapter_config.json")):
        return SFT_ADAPTER, "SFT"
    raise FileNotFoundError("未找到 DPO/SFT adapter，请先完成训练")


def load_model():
    adapter, tag = pick_adapter()
    print(f"使用 {tag} adapter: {adapter}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model, tag


def generate(tokenizer, model, feature_text, instruction):
    user = f"{instruction}\n\n### 音频特征：\n{feature_text}\n\n请给出退化诊断、增强策略和理由。"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=384, do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def check_diagnosis(text, cfg):
    """诊断正确：命中该噪声类型关键词；若含混响/频带限制，也需相应提及。"""
    nt = cfg.get("noise_type", "white")
    ok = any(k in text for k in NOISE_KEYWORDS.get(nt, []))
    if cfg.get("reverb_rt60", 0) and ("混响" not in text and "dereverb" not in text.lower()):
        ok = False
    if cfg.get("bandlimit") and ("频带" not in text and "高频" not in text):
        ok = False
    return ok


PARAM_PATTERNS = [
    r"\d+\s*dB",                 # 衰减 15dB
    r"[Qq]\s*=\s*\d+(\.\d+)?",   # Q=4
    r"\d+\s*[-~]\s*\d+\s*Hz",    # 200-500Hz
    r"\d+\s*Hz",                 # 80Hz
]


def check_format(text):
    """能否解析出至少一个可执行 DSP 参数。"""
    return any(re.search(p, text) for p in PARAM_PATTERNS)


def check_structure(text):
    return ("诊断" in text) and ("策略" in text) and ("理由" in text)


def main():
    with open(METADATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 仅评估带真实退化音频的样本（含 ground-truth 配置），最多 100 条
    eval_items = [d for d in data if d.get("audio_path")][:100]
    print(f"评估样本数: {len(eval_items)}")

    tokenizer, model, tag = load_model()

    diag_ok = fmt_ok = struct_ok = 0
    examples = []
    for i, item in enumerate(eval_items):
        cfg = item["degradation_config"]
        feat = config_to_feature_text(cfg)
        resp = generate(tokenizer, model, feat, "分析这段音频的退化类型并生成增强策略。")
        d, ff, ss = check_diagnosis(resp, cfg), check_format(resp), check_structure(resp)
        diag_ok += d
        fmt_ok += ff
        struct_ok += ss
        if i < 5:
            examples.append({"config": cfg, "response": resp, "diagnosis_ok": d, "format_ok": ff})
        if (i + 1) % 20 == 0:
            print(f"  已评估 {i + 1}/{len(eval_items)}")

    n = max(1, len(eval_items))
    metrics = {
        "adapter": tag,
        "num_eval": len(eval_items),
        "diagnosis_accuracy": round(diag_ok / n, 4),
        "format_accuracy": round(fmt_ok / n, 4),
        "structure_completeness": round(struct_ok / n, 4),
    }
    result = {"metrics": metrics, "examples": examples}
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n========== 评估结果 ==========")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("详细结果已保存:", RESULT)
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
