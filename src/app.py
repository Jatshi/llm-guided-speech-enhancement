"""
Gradio Demo：LLM 引导的语音增强策略生成 + 生产 DSP 执行器。

两种模式：
1) 文本模式：输入音频特征描述 + 用户指令 -> 模型输出退化诊断/增强策略/理由；
2) 音频模式：上传音频 -> 自动提取声学特征 -> 生成策略 -> 安全校验并执行 DSP，返回增强音频。

模型：base(Qwen2.5-1.5B) + GRPO/DPO/SFT LoRA adapter。
在 AutoDL 上通过自定义服务端口 6006 暴露。
"""

import os
import sys

import gradio as gr

# 兼容修复：gradio 5.20 的 gradio_client 在解析 additionalProperties=True(bool) 的
# JSON schema 时会抛 TypeError: argument of type 'bool' is not iterable，导致页面 500。
# 这里对相关函数打补丁，遇到 bool 类型 schema 直接返回 "Any"。
import gradio_client.utils as _gcu
import librosa
import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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

PROJECT = os.environ.get(
    "LSE_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT)
from lse_v2.dsp import ProductionDSPExecutor, plan_from_prescription  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
GRPO_ADAPTER = os.environ.get(
    "GRPO_ADAPTER", os.path.join(PROJECT, "outputs", "native_v4", "grpo", "final", "adapter")
)
DPO_ADAPTER = os.environ.get(
    "DPO_ADAPTER", os.path.join(PROJECT, "outputs", "native_v4", "dpo", "final", "adapter")
)
SFT_ADAPTER = os.environ.get(
    "SFT_ADAPTER", os.path.join(PROJECT, "outputs", "native_v4", "sft", "final", "adapter")
)

SYSTEM_PROMPT = (
    "你是一个专业的语音增强专家，擅长分析音频退化类型、生成可执行的 DSP 增强策略，并解释理由。"
)

_tokenizer = None
_model = None
_tag = None


def _adapter():
    for root, tag in ((GRPO_ADAPTER, "GRPO"), (DPO_ADAPTER, "DPO"), (SFT_ADAPTER, "SFT")):
        direct = os.path.join(root, "adapter_config.json")
        if os.path.exists(direct):
            return root, tag
        if os.path.isdir(root):
            for child in sorted(os.listdir(root)):
                candidate = os.path.join(root, child)
                if os.path.exists(os.path.join(candidate, "adapter_config.json")):
                    return candidate, tag
    raise FileNotFoundError("未找到 GRPO/DPO/SFT adapter，请先完成训练")


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
        out = _model.generate(
            **inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9
        )
    return _tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def extract_features(y, sr):
    """从音频估计一段拟真的声学特征描述（供 LLM 阅读）。"""
    if len(y) < 400:
        return "<audio_analysis>\n- 音频过短，无法分析\n</audio_analysis>"
    # 简易 SNR 估计：能量高/低分位差
    frame = 400
    energies = np.array([np.sum(y[i : i + frame] ** 2) for i in range(0, len(y) - frame, frame)])
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
    lines = [
        "<audio_analysis>",
        f"- 估计信噪比: {snr:.1f}dB",
        f"- 频谱平坦度: {flat:.3f}（{'噪声特征明显' if flat > 0.3 else '语音特征为主'}）",
    ]
    if mid / total > 0.35:
        lines.append("- 频带能量: 200-500Hz 能量偏高（疑似窄带/空调噪声）")
    elif low / total > 0.4:
        lines.append("- 频带能量: 低频(<250Hz)能量偏高")
    else:
        lines.append("- 频带能量: 全频段能量均匀")
    lines.append("</audio_analysis>")
    return "\n".join(lines)


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
    try:
        plan = plan_from_prescription(strategy)
        enhanced = ProductionDSPExecutor().execute(y.astype(np.float32), sr, plan)
    except (ValueError, RuntimeError) as exc:
        return feat, f"处方未通过安全执行器校验：{exc}\n\n原始输出：\n{strategy}", None
    out_path = os.path.join(PROJECT, "outputs", "demo_enhanced.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, enhanced, sr)
    return feat, strategy, out_path


def build_ui():
    with gr.Blocks(title="LLM 引导的语音增强") as demo:
        gr.Markdown(
            f"# LLM 引导的语音增强策略生成\n"
            f"基座 Qwen2.5-1.5B + LoRA（{_tag or 'GRPO/DPO/SFT'}）+ 安全 DSP 执行器"
        )
        with gr.Tab("文本模式"):
            ft = gr.Textbox(
                label="音频特征描述",
                lines=8,
                value=(
                    "<audio_analysis>\n"
                    "- 估计信噪比: 10dB\n"
                    "- 频谱平坦度: 0.45（噪声特征明显）\n"
                    "- 频带能量: 200-500Hz 能量偏高（窄带峰值）\n"
                    "</audio_analysis>"
                ),
            )
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
    build_ui().launch(
        server_name=os.environ.get("LSE_DEMO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("LSE_DEMO_PORT", "6006")),
        share=os.environ.get("LSE_DEMO_SHARE", "false").lower() == "true",
    )
