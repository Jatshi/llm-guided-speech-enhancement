# -*- coding: utf-8 -*-
"""
Gradio Demo：LLM 引导的语音增强策略生成 + 简易谱减增强。

两种模式：
1) 文本模式：输入音频特征描述 + 用户指令 -> 模型输出退化诊断/增强策略/理由；
2) 音频模式：上传音频 -> 自动提取声学特征 -> 生成策略 -> 依据策略做轻量谱减增强，返回增强音频。

模型：base(Qwen2.5-7B) + DPO(或 SFT) LoRA adapter。
在 AutoDL 上通过自定义服务端口 6006 暴露。
"""
import os
import re
import sys
import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr

# 兼容修复：gradio 5.20 的 gradio_client 在解析 additionalProperties=True(bool) 的
# JSON schema 时会抛 TypeError: argument of type 'bool' is not iterable，导致页面 500。
# 这里对相关函数打补丁，遇到 bool 类型 schema 直接返回 "Any"。
import gradio_client.utils as _gcu

_orig_j2p = _gcu._json_schema_to_python_type


def _patched_j2p(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_j2p(schema, defs)


_gcu._json_schema_to_python_type = _patched_j2p

_orig_get_type = _gcu.get_type


def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)


_gcu.get_type = _patched_get_type

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT = os.environ.get("LSE_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT, "models", "Qwen2.5-7B-Instruct"))
DPO_ADAPTER = os.environ.get("DPO_ADAPTER", os.path.join(PROJECT, "outputs", "dpo", "final"))
SFT_ADAPTER = os.environ.get("SFT_ADAPTER", os.path.join(PROJECT, "outputs", "sft", "final"))

SYSTEM_PROMPT = "你是一个专业的语音增强专家，擅长分析音频退化类型、生成可执行的 DSP 增强策略，并解释理由。"

_tokenizer = None
_model = None
_tag = None


def _adapter():
    if os.path.exists(os.path.join(DPO_ADAPTER, "adapter_config.json")):
        return DPO_ADAPTER, "DPO"
    if os.path.exists(os.path.join(SFT_ADAPTER, "adapter_config.json")):
        return SFT_ADAPTER, "SFT"
    raise FileNotFoundError("未找到 DPO/SFT adapter，请先完成训练")


def load_model():
    global _tokenizer, _model, _tag
    if _model is not None:
        return
    adapter, _tag = _adapter()
    print(f"加载 {_tag} adapter: {adapter}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    _model = PeftModel.from_pretrained(base, adapter)
    _model.eval()


def generate_strategy(feature_text, instruction):
    load_model()
    instr = instruction.strip() or "分析这段音频的退化类型并生成增强策略。"
    user = f"{instr}\n\n### 音频特征：\n{feature_text}\n\n请给出退化诊断、增强策略和理由。"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
    return _tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def extract_features(y, sr):
    """从音频估计一段拟真的声学特征描述（供 LLM 阅读）。"""
    if len(y) < 400:
        return "<audio_analysis>\n- 音频过短，无法分析\n</audio_analysis>"
    # 简易 SNR 估计：能量高/低分位差
    frame = 400
    energies = np.array([np.sum(y[i:i + frame] ** 2) for i in range(0, len(y) - frame, frame)])
    energies = energies[energies > 0]
    if len(energies) > 4:
        hi = np.percentile(energies, 90)
        lo = np.percentile(energies, 10) + 1e-9
        snr = float(np.clip(10 * np.log10(hi / lo), 3, 30))
    else:
        snr = 15.0
    S = np.abs(librosa.stft(y, n_fft=512))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    band = S.mean(axis=1)
    low = band[freqs < 250].sum()
    mid = band[(freqs >= 200) & (freqs < 500)].sum()
    total = band.sum() + 1e-9
    lines = ["<audio_analysis>", f"- 估计信噪比: {snr:.1f}dB",
             f"- 频谱平坦度: {flat:.3f}（{'噪声特征明显' if flat > 0.3 else '语音特征为主'}）"]
    if mid / total > 0.35:
        lines.append("- 频带能量: 200-500Hz 能量偏高（疑似窄带/空调噪声）")
    elif low / total > 0.4:
        lines.append("- 频带能量: 低频(<250Hz)能量偏高")
    else:
        lines.append("- 频带能量: 全频段能量均匀")
    lines.append("</audio_analysis>")
    return "\n".join(lines)


def spectral_subtraction(y, sr, strength=1.0):
    """轻量谱减降噪：用前 0.3s 估计噪声谱后做减法。"""
    n_fft, hop = 512, 128
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    noise_frames = max(1, int(0.3 * sr / hop))
    noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    clean_mag = np.maximum(mag - strength * noise_mag, 0.0)
    out = librosa.istft(clean_mag * np.exp(1j * phase), hop_length=hop, length=len(y))
    m = np.max(np.abs(out))
    return out / m * 0.95 if m > 0 else out


def parse_strength(strategy_text):
    """从策略文本粗略解析降噪强度：重度>轻度；带阻/衰减越大越强。"""
    s = 1.0
    if "重度谱减" in strategy_text:
        s = 1.6
    elif "轻度谱减" in strategy_text:
        s = 0.8
    m = re.search(r"衰减\s*(\d+)\s*dB", strategy_text)
    if m:
        s = float(np.clip(int(m.group(1)) / 15.0, 0.5, 2.0))
    return s


def run_text(feature_text, instruction):
    if not feature_text.strip():
        return "请输入音频特征描述（<audio_analysis> 块）。"
    return generate_strategy(feature_text, instruction)


def run_audio(audio_path, instruction):
    if audio_path is None:
        return "请先上传音频。", "", None
    y, sr = librosa.load(audio_path, sr=16000, duration=10.0)
    feat = extract_features(y, sr)
    strategy = generate_strategy(feat, instruction)
    enhanced = spectral_subtraction(y, sr, strength=parse_strength(strategy))
    out_path = os.path.join(PROJECT, "outputs", "demo_enhanced.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, enhanced, sr)
    return feat, strategy, out_path


def build_ui():
    with gr.Blocks(title="LLM 引导的语音增强") as demo:
        gr.Markdown(f"# LLM 引导的语音增强策略生成\n基座 Qwen2.5-7B + LoRA（{_tag or 'SFT/DPO'}）")
        with gr.Tab("文本模式"):
            ft = gr.Textbox(label="音频特征描述", lines=8,
                            value="<audio_analysis>\n- 估计信噪比: 10dB\n- 频谱平坦度: 0.45（噪声特征明显）\n- 频带能量: 200-500Hz 能量偏高（窄带峰值）\n</audio_analysis>")
            it = gr.Textbox(label="用户指令（可选）", value="去掉空调声，保留人声")
            bt = gr.Button("生成增强策略", variant="primary")
            ot = gr.Textbox(label="模型输出（诊断 / 策略 / 理由）", lines=16)
            bt.click(run_text, [ft, it], ot)
        with gr.Tab("音频模式"):
            au = gr.Audio(label="上传音频", type="filepath")
            ia = gr.Textbox(label="用户指令（可选）", value="去除背景噪声，保留人声自然度")
            ba = gr.Button("分析并增强", variant="primary")
            of = gr.Textbox(label="自动提取的音频特征", lines=6)
            os_ = gr.Textbox(label="增强策略", lines=14)
            oa = gr.Audio(label="增强后音频")
            ba.click(run_audio, [au, ia], [of, os_, oa])
    return demo


if __name__ == "__main__":
    load_model()
    build_ui().launch(server_name="0.0.0.0", server_port=6006, share=True)
