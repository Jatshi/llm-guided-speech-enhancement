"""Continuous audio-latent interface and a deterministic complex-STFT backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class STFTCodecConfig:
    n_fft: int = 512
    hop_length: int = 128
    win_length: int | None = None
    normalized: bool = True
    scale: float = 16.0


class ComplexSTFTCodec:
    """Invertible continuous representation used until a frozen Audio VAE is selected."""

    latent_channels = 2

    def __init__(self, config: STFTCodecConfig | None = None) -> None:
        self.config = config or STFTCodecConfig()
        if self.config.n_fft < 16 or self.config.hop_length < 1:
            raise ValueError("invalid STFT configuration")
        if self.config.scale <= 0:
            raise ValueError("STFT latent scale must be positive")

    def _window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        length = self.config.win_length or self.config.n_fft
        return torch.hann_window(length, device=device, dtype=dtype)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        if not waveform.is_floating_point():
            waveform = waveform.float()
        spectrum = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window(waveform.device, waveform.dtype),
            center=True,
            normalized=self.config.normalized,
            return_complex=True,
        )
        return torch.stack((spectrum.real, spectrum.imag), dim=1) * self.config.scale

    def decode(self, latent: torch.Tensor, *, length: int) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != 2:
            raise ValueError("latent must have shape [batch, 2, frequency, time]")
        latent = latent / self.config.scale
        spectrum = torch.complex(latent[:, 0], latent[:, 1])
        return torch.istft(
            spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window(latent.device, latent.dtype),
            center=True,
            normalized=self.config.normalized,
            length=length,
        )
