# -*- coding: utf-8 -*-
"""
阶段 1：SFT（监督微调）—— Qwen2.5-7B-Instruct + LoRA。

相对 V2 原稿的工程修正：
- 使用本地模型路径（ModelScope 下载），避免联网；
- report_to 仅 tensorboard（不依赖 wandb 登录）；
- 动态 padding（DataCollatorForSeq2Seq），不再把每条样本都 pad 到 2048，大幅提速；
- 不保存 15GB 合并模型（推理用 base+adapter），规避磁盘不足。
"""
import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ==================== 配置 ====================
_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_ROOT, "models/Qwen2.5-7B-Instruct"))
DATA_DIR = os.path.join(_ROOT, "data/training/llm_format")
OUTPUT_DIR = os.path.join(_ROOT, "outputs/sft")
LOG_DIR = os.path.join(_ROOT, "outputs/logs/sft_tb")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 2
GRAD_ACCUM = 8            # 等效 batch=16
NUM_EPOCHS = 1           # 1 个 epoch（108K 样本足够）：约 6750 步 ~5.9h，压缩总时长到 ~8h
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024     # 样本较短，1024 足够
WARMUP_RATIO = 0.03
SAVE_STEPS = 2000        # 仅保存少量检查点（用于崩溃恢复），减少磁盘占用
EVAL_STEPS = 500
LOGGING_STEPS = 20
SAVE_TOTAL_LIMIT = 1     # 只保留最新 1 个检查点（autodl-tmp 仅 ~4.6G）


def load_split(split):
    with open(os.path.join(DATA_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
        return Dataset.from_list(json.load(f))


def make_preprocess(tokenizer):
    def _fn(examples):
        input_ids_list, labels_list, attn_list = [], [], []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer(text, max_length=MAX_SEQ_LENGTH, truncation=True)["input_ids"]
            input_ids_list.append(ids)
            labels_list.append(list(ids))          # 全序列 LM 损失
            attn_list.append([1] * len(ids))
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attn_list}
    return _fn


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    print("设备:", torch.cuda.get_device_name(0))
    print("显存:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

    print("加载 tokenizer / 模型...", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = load_split("train")
    eval_ds = load_split("eval")
    print(f"训练样本: {len(train_ds)}  验证样本: {len(eval_ds)}")

    pre = make_preprocess(tokenizer)
    train_ds = train_ds.map(pre, batched=True, remove_columns=train_ds.column_names, num_proc=4, desc="tokenize-train")
    eval_ds = eval_ds.map(pre, batched=True, remove_columns=eval_ds.column_names, num_proc=4, desc="tokenize-eval")

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        eval_strategy="no",     # 关闭周期性评估（12000 条 eval 每次约 11min，共约 2.4h 开销）；最终由 evaluate.py 生成指标
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, tokenizer=tokenizer,
    )

    steps = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS
    print(f"预计训练步数: ~{steps}")
    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("SFT LoRA adapter 已保存:", final_path)


if __name__ == "__main__":
    main()
