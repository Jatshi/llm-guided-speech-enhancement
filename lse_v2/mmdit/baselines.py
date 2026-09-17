"""Strong external enhancement baselines loaded only when explicitly configured."""

from __future__ import annotations

import torch

from .data import _resample


class DeepFilterNetBaseline:
    """Official DeepFilterNet inference adapter with sample-rate round trips."""

    def __init__(self, model_name: str = "DeepFilterNet3") -> None:
        try:
            from df.enhance import init_df
        except ImportError as exc:
            raise RuntimeError(
                "DeepFilterNet baseline requested but deepfilternet is not installed"
            ) from exc
        loaded = init_df(
            model_base_dir=model_name,
            log_file=None,
            config_allow_defaults=True,
        )
        self.model, self.state = loaded[0], loaded[1]
        self.sample_rate = int(self.state.sr())

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        from df.enhance import enhance

        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("DeepFilterNet adapter expects mono [1, samples] audio")
        cpu = waveform.detach().float().cpu().squeeze(0)
        converted = _resample(cpu, sample_rate, self.sample_rate).unsqueeze(0)
        enhanced = enhance(self.model, self.state, converted, pad=True)
        if enhanced.ndim == 2:
            enhanced = enhanced.squeeze(0)
        restored = _resample(enhanced.float().cpu(), self.sample_rate, sample_rate)
        target_length = waveform.shape[-1]
        if restored.numel() < target_length:
            restored = torch.nn.functional.pad(restored, (0, target_length - restored.numel()))
        return restored[:target_length].view(1, -1).to(waveform.device)
