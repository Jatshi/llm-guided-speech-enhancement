# -*- coding: utf-8 -*-
"""
音频退化模拟器 + 特征文本描述（V2，磁盘友好版）。

设计要点：
- SFT/DPO 训练只需要"退化配置"文本，不需要真实退化音频，因此批量生成时不落盘音频；
- 仅对少量样本（用于评估/Demo）真正生成并保存退化 wav；
- 复用 V1 中较完整的噪声/混响/频带限制实现，保证 Demo 端到端可跑。
"""
import numpy as np
import random
from scipy.signal import butter, filtfilt


class NoiseGenerator:
    """程序生成各类噪声"""

    @staticmethod
    def white_noise(n, sr=16000):
        return np.random.randn(n)

    @staticmethod
    def pink_noise(n, sr=16000):
        white = np.random.randn(n)
        x = np.fft.rfft(white)
        f = np.sqrt(np.arange(1, len(x) + 1))
        pink = np.fft.irfft(x / f, n=n)
        m = np.max(np.abs(pink))
        return pink / m if m > 0 else pink

    @staticmethod
    def hvac_noise(n, sr=16000):
        t = np.arange(n) / sr
        noise = np.zeros(n)
        for fr in [220, 280, 350, 420, 480]:
            noise += np.sin(2 * np.pi * fr * t) * random.uniform(0.5, 1.0)
        noise += 0.3 * np.random.randn(n)
        b, a = butter(4, [200 / (sr / 2), 500 / (sr / 2)], btype="band")
        noise = filtfilt(b, a, noise)
        m = np.max(np.abs(noise))
        return noise / m if m > 0 else noise

    @staticmethod
    def cafe_noise(n, sr=16000):
        t = np.arange(n) / sr
        noise = 0.5 * np.sin(2 * np.pi * 100 * t)
        for fr in range(500, 4000, 200):
            mod = np.sin(2 * np.pi * random.uniform(2, 8) * t)
            noise += random.uniform(0.1, 0.3) * mod * np.sin(2 * np.pi * fr * t)
        noise += 0.2 * np.random.randn(n)
        m = np.max(np.abs(noise))
        return noise / m if m > 0 else noise


class Degradation:
    """退化管道：加混响 -> 加噪声 -> 频带限制"""

    def __init__(self, sr=16000):
        self.sr = sr
        self.ng = NoiseGenerator()

    def add_reverb(self, audio, rt60):
        n = int(rt60 * self.sr * 2)
        if n < 10:
            return audio
        ir = np.exp(-np.arange(n) / (rt60 * self.sr / 6.91))
        ir = ir / np.sum(ir)
        wet = np.convolve(audio, ir, mode="same")
        mixed = 0.6 * audio + 0.4 * wet
        m = np.max(np.abs(mixed))
        return mixed / m if m > 0 else mixed

    def add_noise(self, audio, noise_type, snr_db):
        n = len(audio)
        gen = {
            "white": self.ng.white_noise,
            "pink": self.ng.pink_noise,
            "hvac": self.ng.hvac_noise,
            "cafe": self.ng.cafe_noise,
        }.get(noise_type, self.ng.white_noise)
        noise = gen(n, self.sr)
        sp = np.mean(audio ** 2)
        npow = np.mean(noise ** 2) + 1e-12
        scale = np.sqrt(sp / (npow * (10 ** (snr_db / 10))))
        return audio + scale * noise

    def bandlimit(self, audio, low, high):
        b, a = butter(4, [low / (self.sr / 2), high / (self.sr / 2)], btype="band")
        return filtfilt(b, a, audio)

    def apply(self, audio, cfg):
        out = audio.copy()
        if cfg.get("reverb_rt60", 0):
            out = self.add_reverb(out, cfg["reverb_rt60"])
        out = self.add_noise(out, cfg.get("noise_type", "white"), cfg.get("snr_db", 15))
        if cfg.get("bandlimit"):
            lo, hi = cfg["bandlimit"]
            out = self.bandlimit(out, lo, hi)
        m = np.max(np.abs(out))
        return out / m * 0.95 if m > 0 else out


def sample_config(rng: random.Random):
    """随机采样一个退化配置（仅配置，不涉及音频）"""
    cfg = {
        "noise_type": rng.choice(["white", "pink", "hvac", "cafe"]),
        "snr_db": round(rng.uniform(5, 25), 1),
    }
    if rng.random() < 0.5:
        cfg["reverb_rt60"] = round(rng.uniform(0.1, 0.6), 2)
    if rng.random() < 0.2:
        cfg["bandlimit"] = [300, 3400]
    return cfg


# 退化配置 -> 供 LLM 阅读的"音频特征描述"文本
NOISE_ZH = {"white": "白噪声", "pink": "粉红噪声", "hvac": "空调噪声", "cafe": "咖啡厅噪声"}


def config_to_feature_text(cfg):
    """把退化配置渲染为一段拟真的音频特征描述（作为 LLM 输入的一部分）"""
    nt = cfg.get("noise_type", "white")
    snr = cfg.get("snr_db", 15)
    rt60 = cfg.get("reverb_rt60", 0)
    band = cfg.get("bandlimit")
    lines = ["<audio_analysis>"]
    lines.append(f"- 估计信噪比: {snr}dB")
    # 频谱平坦度：噪声越强越平坦
    flat = round(min(0.6, 0.15 + (25 - snr) / 40.0), 3)
    lines.append(f"- 频谱平坦度: {flat}（{'噪声特征明显' if flat > 0.3 else '语音特征为主'}）")
    # 频带能量提示
    if nt == "hvac":
        lines.append("- 频带能量: 200-500Hz 能量偏高（窄带峰值）")
    elif nt == "pink":
        lines.append("- 频带能量: 低频(<250Hz)能量偏高")
    elif nt == "cafe":
        lines.append("- 频带能量: 中高频(500-4000Hz)存在背景 chatter")
    else:
        lines.append("- 频带能量: 全频段能量均匀升高")
    if rt60:
        lines.append(f"- 时域拖尾: 存在混响拖尾，估计 RT60≈{rt60}s")
    if band:
        lines.append(f"- 有效带宽: 约 {band[0]}-{band[1]}Hz（高频缺失）")
    lines.append("</audio_analysis>")
    return "\n".join(lines)
