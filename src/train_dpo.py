# -*- coding: utf-8 -*-
"""
阶段 2：DPO（直接偏好优化）—— 在 SFT LoRA 基础上继续训练。

工程要点（适配 trl 0.12 / 显存与磁盘）：
- 使用 DPOConfig（beta/max_length 等参数已迁移到 config）；
- 采用对话式（conversational）数据格式，prompt/chosen/rejected 均为 messages 列表；
- 加载 base + SFT adapter（is_trainable=True）作为 policy，ref 由禁用 adapter 自动得到，省显存；
- rejected（坏策略）通过多点扰动构造：过度衰减、错误 Q、重度谱减、矛盾操作等。
"""
import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

random.seed(42)

_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_ROOT, "models/Qwen2.5-7B-Instruct"))
SFT_ADAPTER = os.path.join(_ROOT, "outputs/sft/final")
SFT_TRAIN = os.path.join(_ROOT, "data/training/llm_format/train.json")
DPO_DATA = os.path.join(_ROOT, "data/training/dpo_data.json")
OUTPUT_DIR = os.path.join(_ROOT, "outputs/dpo")
LOG_DIR = os.path.join(_ROOT, "outputs/logs/dpo_tb")

NUM_PAIRS = 10000
BATCH_SIZE = 2
GRAD_ACCUM = 8
NUM_EPOCHS = 2
LEARNING_RATE = 5e-5
BETA = 0.1


def corrupt(good: str) -> str:
    """把"好策略"扰动为"坏策略"：过度处理 / 错误参数 / 矛盾操作。"""
    bad = good
    bad = bad.replace("衰减 15dB", "衰减 35dB")        # 过度衰减
    bad = bad.replace("Q=4", "Q=0.3")                   # 不合理 Q 值（过宽）
    bad = bad.replace("轻度谱减", "重度谱减")            # 过度处理
    bad = bad.replace("80Hz 以下衰减", "800Hz 以下衰减")  # 误伤人声基频
    bad = bad.replace("保持人声自然度", "同时对全频段强压制")  # 破坏自然度
    # 追加一条明显有害的矛盾操作
    if "增强策略：" in bad:
        bad = bad.replace("增强策略：", "增强策略：\n0. 全频段限幅压缩 20dB，忽略退化类型直接暴力降噪")
    if bad == good:
        bad = good + "\n补充：直接对全频段做重度谱减并最大化增益。"
    return bad


def build_dpo_dataset():
    with open(SFT_TRAIN, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in random.sample(data, min(NUM_PAIRS, len(data))):
        m = item["messages"]
        prompt = m[:2]                       # system + user
        good = m[2]["content"]
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": good}],
            "rejected": [{"role": "assistant", "content": corrupt(good)}],
        })
    with open(DPO_DATA, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"DPO 数据: {len(pairs)} 对 -> {DPO_DATA}")
    return pairs


def main():
    pairs = build_dpo_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载 base + SFT adapter ...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()

    ds = Dataset.from_list(pairs)
    split = int(0.95 * len(ds))
    train_ds = ds.select(range(split))
    eval_ds = ds.select(range(split, len(ds)))

    args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=1024,
        max_prompt_length=768,
        logging_dir=LOG_DIR,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        report_to=["tensorboard"],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,               # PEFT：禁用 adapter 得到参考模型，省显存
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("开始 DPO 训练...")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print("DPO adapter 已保存:", os.path.join(OUTPUT_DIR, "final"))


if __name__ == "__main__":
    main()
