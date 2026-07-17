# -*- coding: utf-8 -*-
"""
构建 LLM SFT 训练数据（train.json / eval.json）。

从 metadata.json 的退化配置出发，程序化生成"退化诊断 + 增强策略 + 理由"的高质量回答，
并配合多样化的用户指令，组成对话式训练样本。
"""
import os
import sys
import json
import random
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degradation import config_to_feature_text, NOISE_ZH

random.seed(42)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get("LSE_TRAINING_DIR", os.path.join(_ROOT, "data", "training"))

SYSTEM_PROMPT = "你是一个专业的语音增强专家，擅长分析音频退化类型、生成可执行的 DSP 增强策略，并解释理由。"

USER_TEMPLATES = [
    "分析这段音频的退化类型并生成增强策略。",
    "这段音频有什么问题？请给出修复方案。",
    "我想清理这段音频，请告诉我具体怎么做。",
    "检测音频中的噪声类型并给出处理建议。",
    "分析音频质量并生成增强参数。",
    "用户指令：去掉背景里的噪声。",
    "用户指令：保留人声的自然感，只去除噪声。",
    "用户指令：让声音更清晰一些。",
    "用户指令：去掉低频嗡嗡声，保留人声温暖感。",
    "用户指令：去除混响和背景噪声。",
]


def build_response(cfg):
    """根据退化配置构建标准（chosen）回答：诊断 / 策略 / 理由"""
    nt = cfg.get("noise_type", "white")
    snr = cfg.get("snr_db", 15)
    rt60 = cfg.get("reverb_rt60", 0)
    band = cfg.get("bandlimit")
    zh = NOISE_ZH.get(nt, nt)

    diag, strat, reason = [], [], []
    diag.append("退化诊断：")
    n = 1
    if nt == "hvac":
        diag.append(f"{n}. 窄带噪声：200-500Hz，空调压缩机噪声特征，信噪比约 {snr}dB")
    elif nt == "white":
        diag.append(f"{n}. 宽带白噪声：全频段均匀分布，信噪比约 {snr}dB")
    elif nt == "pink":
        diag.append(f"{n}. 粉红噪声：低频能量偏高，类似风扇声，信噪比约 {snr}dB")
    else:
        diag.append(f"{n}. 背景人声噪声：中高频 chatter 特征，信噪比约 {snr}dB")
    n += 1
    if rt60:
        diag.append(f"{n}. 房间混响：RT60 ≈ {rt60}s，{'轻微' if rt60 < 0.4 else '明显'}混响")
        n += 1
    if band:
        diag.append(f"{n}. 频带限制：{band[0]}-{band[1]}Hz，电话音质特征")

    strat.append("增强策略：")
    if nt == "hvac":
        strat.append("1. 带阻滤波：200-500Hz，衰减 15dB，Q=4")
        strat.append("2. 谱减法：在 200-500Hz 频段应用轻度谱减")
        strat.append("3. 保留：500Hz 以上频段不处理，保持人声自然度")
    elif nt == "white":
        strat.append("1. 谱减法：全频段应用轻度谱减")
        strat.append("2. 维纳滤波：估计噪声谱后进行维纳滤波")
        strat.append(f"3. 自适应阈值：根据 SNR={snr}dB 调整降噪强度")
    elif nt == "pink":
        strat.append("1. 高通滤波：80Hz 以下衰减，去除低频轰鸣")
        strat.append("2. 谱减法：在 0-500Hz 频段应用轻度谱减")
        strat.append("3. 动态均衡：补偿低频过度衰减")
    else:
        strat.append("1. 自适应降噪：针对中高频背景人声")
        strat.append("2. 谱减法：在 500-4000Hz 频段应用轻度谱减")
        strat.append("3. 语音增强：提升 1000-3000Hz 频段清晰度")
    k = 4
    if rt60:
        strat.append(f"{k}. 去混响：RT60={rt60}s 的 dereverb 处理")
        k += 1
    if band:
        strat.append(f"{k}. 频带扩展：使用谐波恢复扩展 {band[1]}Hz 以上高频")

    reason.append("理由：")
    if nt == "hvac":
        reason.append("空调噪声主要在 200-500Hz 呈窄带分布，与人声基频不重叠，带阻滤波可有效去除而不影响可懂度。")
    elif nt == "white":
        reason.append("白噪声全频段均匀分布，谱减法结合维纳滤波可在频域实现最小均方误差估计。")
    elif nt == "pink":
        reason.append("粉红噪声能量集中在低频，高通滤波与低频谱减可去除轰鸣同时保留主要人声频段。")
    else:
        reason.append("咖啡厅噪声以中高频背景人声为主，与目标语音频谱重叠，需自适应谱减与语音增强联合处理。")
    if rt60:
        reason.append(f"检测到混响时间 {rt60}s，使用对应 RT60 的 dereverb 去除房间反射。")
    if band:
        reason.append(f"检测到 {band[1]}Hz 以上高频缺失，做频带扩展恢复自然度。")

    return "\n".join(diag) + "\n\n" + "\n".join(strat) + "\n\n" + "\n".join(reason)


def build_sample(rec):
    cfg = rec["degradation_config"]
    instruction = random.choice(USER_TEMPLATES)
    feat = config_to_feature_text(cfg)
    user = f"{instruction}\n\n### 音频特征：\n{feat}\n\n请给出退化诊断、增强策略和理由。"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": build_response(cfg)},
    ]
    return {"messages": messages}


def main():
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"总样本数: {len(data)}")

    samples = [build_sample(r) for r in tqdm(data, desc="构建样本")]
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train, ev = samples[:split], samples[split:]

    out = os.path.join(OUT_DIR, "llm_format")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False)
    with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False)
    print(f"训练集: {len(train)} 条  验证集: {len(ev)} 条  -> {out}")
    print("样本示例:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
