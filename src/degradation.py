"""Synthetic degradations retained for the demo and legacy data generator."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

NOISE_ZH = {
    "white": "白噪声",
    "pink": "粉红噪声",
    "hvac": "空调噪声",
    "cafe": "咖啡厅背景声",
}


class NoiseGenerator:
    @staticmethod
    def white_noise(length: int, sample_rate: int = 16000) -> np.ndarray:
        del sample_rate
        return np.random.randn(length)

    @staticmethod
    def pink_noise(length: int, sample_rate: int = 16000) -> np.ndarray:
        del sample_rate
        spectrum = np.fft.rfft(np.random.randn(length))
        scale = np.sqrt(np.arange(1, len(spectrum) + 1))
        signal = np.fft.irfft(spectrum / scale, n=length)
        peak = np.max(np.abs(signal))
        return signal / peak if peak else signal

    @staticmethod
    def hvac_noise(length: int, sample_rate: int = 16000) -> np.ndarray:
        time = np.arange(length) / sample_rate
        signal = sum(
            np.sin(2 * np.pi * frequency * time) for frequency in (220, 280, 350, 420, 480)
        )
        signal = signal + 0.3 * np.random.randn(length)
        low, high = 200 / (sample_rate / 2), 500 / (sample_rate / 2)
        b, a = butter(4, [low, high], btype="band")
        signal = filtfilt(b, a, signal)
        peak = np.max(np.abs(signal))
        return signal / peak if peak else signal

    @staticmethod
    def cafe_noise(length: int, sample_rate: int = 16000) -> np.ndarray:
        time = np.arange(length) / sample_rate
        signal = np.zeros(length)
        for frequency in range(500, 4000, 200):
            modulation = np.sin(2 * np.pi * random.uniform(2, 8) * time)
            signal += 0.2 * modulation * np.sin(2 * np.pi * frequency * time)
        signal += 0.2 * np.random.randn(length)
        peak = np.max(np.abs(signal))
        return signal / peak if peak else signal


class Degradation:
    def __init__(self, sr: int = 16000) -> None:
        self.sr = sr
        self.noise = NoiseGenerator()

    def add_reverb(self, audio: np.ndarray, rt60: float) -> np.ndarray:
        length = max(10, int(rt60 * self.sr * 2))
        impulse = np.exp(-np.arange(length) / max(1, rt60 * self.sr / 6.91))
        impulse /= np.sum(impulse)
        wet = np.convolve(audio, impulse, mode="same")
        return 0.6 * audio + 0.4 * wet

    def add_noise(self, audio: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
        generator = {
            "white": self.noise.white_noise,
            "pink": self.noise.pink_noise,
            "hvac": self.noise.hvac_noise,
            "cafe": self.noise.cafe_noise,
        }.get(noise_type, self.noise.white_noise)
        noise = generator(len(audio), self.sr)
        signal_power = np.mean(audio**2)
        noise_power = np.mean(noise**2) + 1e-12
        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        return audio + scale * noise

    def bandlimit(self, audio: np.ndarray, low: float, high: float) -> np.ndarray:
        b, a = butter(4, [low / (self.sr / 2), high / (self.sr / 2)], btype="band")
        return filtfilt(b, a, audio)

    def apply(self, audio: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        output = np.asarray(audio, dtype=np.float32).copy()
        if config.get("reverb_rt60"):
            output = self.add_reverb(output, float(config["reverb_rt60"]))
        output = self.add_noise(
            output,
            str(config.get("noise_type", "white")),
            float(config.get("snr_db", 15)),
        )
        if config.get("bandlimit"):
            output = self.bandlimit(output, *config["bandlimit"])
        peak = np.max(np.abs(output))
        return output / peak * 0.95 if peak else output


def sample_config(rng: random.Random) -> dict[str, Any]:
    config: dict[str, Any] = {
        "noise_type": rng.choice(list(NOISE_ZH)),
        "snr_db": round(rng.uniform(5, 25), 1),
    }
    if rng.random() < 0.5:
        config["reverb_rt60"] = round(rng.uniform(0.1, 0.6), 2)
    if rng.random() < 0.2:
        config["bandlimit"] = [300, 3400]
    return config


def config_to_feature_text(config: dict[str, Any]) -> str:
    return (
        f"noise_type={config.get('noise_type', 'unknown')}; "
        f"snr_db={config.get('snr_db', 'unknown')}; "
        f"reverb_rt60={config.get('reverb_rt60', 'none')}; "
        f"bandlimit_hz={config.get('bandlimit', 'none')}"
    )
