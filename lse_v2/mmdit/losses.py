"""Perceptual and multi-resolution objectives for speech-preserving enhancement."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Compare waveform spectra at complementary time-frequency resolutions."""

    def __init__(
        self,
        *,
        fft_sizes: Sequence[int] = (256, 512, 1024),
        hop_sizes: Sequence[int] = (64, 128, 256),
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        if len(fft_sizes) != len(hop_sizes) or not fft_sizes:
            raise ValueError("fft_sizes and hop_sizes must be non-empty and equally sized")
        if any(fft < 16 for fft in fft_sizes) or any(hop < 1 for hop in hop_sizes):
            raise ValueError("invalid STFT resolution")
        if any(hop > fft for fft, hop in zip(fft_sizes, hop_sizes, strict=True)):
            raise ValueError("each hop size must not exceed its FFT size")
        self.fft_sizes = tuple(int(value) for value in fft_sizes)
        self.hop_sizes = tuple(int(value) for value in hop_sizes)
        self.epsilon = float(epsilon)

    @staticmethod
    def _waveform(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples] or [batch, 1, samples]")
        return value.float()

    @staticmethod
    def _magnitude(value: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        window = torch.hann_window(n_fft, device=value.device, dtype=value.dtype)
        return torch.stft(
            value,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        estimate = self._waveform(estimate)
        reference = self._waveform(reference)
        if estimate.shape != reference.shape:
            raise ValueError("estimate and reference shapes must match")
        terms = []
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes, strict=True):
            estimate_mag = self._magnitude(estimate, n_fft, hop)
            reference_mag = self._magnitude(reference, n_fft, hop)
            spectral_convergence = torch.linalg.vector_norm(estimate_mag - reference_mag) / (
                torch.linalg.vector_norm(reference_mag) + self.epsilon
            )
            log_magnitude = F.l1_loss(
                torch.log(estimate_mag + self.epsilon),
                torch.log(reference_mag + self.epsilon),
            )
            terms.append(spectral_convergence + log_magnitude)
        return torch.stack(terms).mean()


def _teacher_features(output: Any, layer: int | None) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        features = output
    elif layer is not None and getattr(output, "hidden_states", None):
        features = output.hidden_states[layer]
    elif getattr(output, "last_hidden_state", None) is not None:
        features = output.last_hidden_state
    elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        features = output[0]
    else:
        raise TypeError("semantic teacher must return a tensor or last_hidden_state")
    if features.ndim == 2:
        features = features.unsqueeze(-1)
    if features.ndim != 3:
        raise ValueError("semantic features must have shape [batch, frames, channels]")
    return features.float()


def _normalize_waveform(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    centered = value - value.mean(dim=-1, keepdim=True)
    return centered / centered.std(dim=-1, keepdim=True).clamp_min(1e-5)


def semantic_consistency_loss(
    teacher: nn.Module,
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    layer: int | None = None,
) -> torch.Tensor:
    """Preserve linguistic content using a frozen self-supervised audio teacher.

    Teacher parameters are frozen here rather than relying on the caller.  The
    estimate path remains differentiable so gradients reach the enhancer input.
    """

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    estimate_input = _normalize_waveform(estimate)
    reference_input = _normalize_waveform(reference)
    estimate_features = _teacher_features(
        teacher(estimate_input, output_hidden_states=layer is not None)
        if _accepts_hidden_state_flag(teacher)
        else teacher(estimate_input),
        layer,
    )
    with torch.no_grad():
        reference_features = _teacher_features(
            teacher(reference_input, output_hidden_states=layer is not None)
            if _accepts_hidden_state_flag(teacher)
            else teacher(reference_input),
            layer,
        )
    frames = min(estimate_features.shape[1], reference_features.shape[1])
    channels = min(estimate_features.shape[2], reference_features.shape[2])
    estimate_features = estimate_features[:, :frames, :channels]
    reference_features = reference_features[:, :frames, :channels]
    similarity = F.cosine_similarity(estimate_features, reference_features, dim=-1)
    return (1 - similarity).mean()


def _accepts_hidden_state_flag(teacher: nn.Module) -> bool:
    """Avoid imposing the Transformers call signature on lightweight test teachers."""

    module = teacher.forward
    code = getattr(module, "__code__", None)
    return bool(code and "output_hidden_states" in code.co_varnames)


def load_frozen_semantic_teacher(model_name: str, device: torch.device) -> nn.Module:
    """Load a Hugging Face audio encoder lazily on the paid-GPU environment."""

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required for semantic consistency training") from exc
    teacher = AutoModel.from_pretrained(model_name, trust_remote_code=False).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher
