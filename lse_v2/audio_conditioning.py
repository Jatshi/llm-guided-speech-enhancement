"""Native audio-prefix projection for an audio-conditioned language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class AudioConditioningConfig:
    encoder_dim: int
    llm_dim: int
    pooling_stride: int = 4
    prefix_tokens: int | None = None
    freeze_encoder: bool = True

    def __post_init__(self) -> None:
        if self.encoder_dim <= 0 or self.llm_dim <= 0:
            raise ValueError("encoder_dim and llm_dim must be positive")
        if self.pooling_stride <= 0:
            raise ValueError("pooling_stride must be positive")
        if self.prefix_tokens is not None and self.prefix_tokens <= 0:
            raise ValueError("prefix_tokens must be positive when configured")


class AudioPrefixProjector(nn.Module):
    """Pool encoded audio to a fixed prefix and map it to the LLM width."""

    def __init__(self, config: AudioConditioningConfig) -> None:
        super().__init__()
        self.config = config
        self.projector = nn.Sequential(
            nn.LayerNorm(config.encoder_dim),
            nn.Linear(config.encoder_dim, config.llm_dim),
            nn.GELU(),
            nn.Linear(config.llm_dim, config.llm_dim),
        )

    def forward(self, encoded: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        if encoded.ndim != 3 or encoded.shape[-1] != self.config.encoder_dim:
            raise ValueError("encoded audio must have shape [batch, frames, encoder_dim]")
        if self.config.prefix_tokens is None:
            pooled = F.avg_pool1d(
                encoded.transpose(1, 2),
                kernel_size=self.config.pooling_stride,
                stride=self.config.pooling_stride,
                ceil_mode=True,
            ).transpose(1, 2)
            return self.projector(pooled)
        if frame_mask is None:
            frame_mask = encoded.new_ones(encoded.shape[:2], dtype=encoded.dtype)
        if frame_mask.ndim != 2 or frame_mask.shape[0] != encoded.shape[0]:
            raise ValueError("frame_mask must have shape [batch, frames]")
        if frame_mask.shape[1] != encoded.shape[1]:
            frame_mask = F.interpolate(
                frame_mask.to(dtype=encoded.dtype).unsqueeze(1),
                size=encoded.shape[1],
                mode="nearest",
            ).squeeze(1)
        rows: list[Tensor] = []
        for row, mask in zip(encoded, frame_mask, strict=True):
            valid = row[mask.to(dtype=torch.bool)]
            if valid.shape[0] == 0:
                raise ValueError("each audio example requires at least one valid frame")
            pooled = F.adaptive_avg_pool1d(
                valid.transpose(0, 1).unsqueeze(0), self.config.prefix_tokens
            ).transpose(1, 2)
            rows.append(pooled.squeeze(0))
        return self.projector(torch.stack(rows, dim=0))


class WhisperAudioProjector(nn.Module):
    """Map Whisper-like hidden states into continuous LLM prefix embeddings.

    The caller owns token insertion and the causal language-model loss.  Keeping
    that boundary explicit makes it possible to swap Qwen/Llama-family decoders
    without coupling the audio encoder to one Transformers implementation.
    """

    def __init__(self, encoder: nn.Module, config: AudioConditioningConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.projector = nn.Sequential(
            nn.LayerNorm(config.encoder_dim),
            nn.Linear(config.encoder_dim, config.llm_dim),
            nn.GELU(),
            nn.Linear(config.llm_dim, config.llm_dim),
        )
        if config.freeze_encoder:
            self.encoder.requires_grad_(False)

    def forward(self, input_features: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [batch, frames, features]")
        encoded = self.encoder(input_features).last_hidden_state
        if encoded.ndim != 3 or encoded.shape[-1] != self.config.encoder_dim:
            raise ValueError("audio encoder output must have shape [batch, frames, encoder_dim]")
        if self.config.prefix_tokens is None:
            pooled = F.avg_pool1d(
                encoded.transpose(1, 2),
                kernel_size=self.config.pooling_stride,
                stride=self.config.pooling_stride,
                ceil_mode=True,
            ).transpose(1, 2)
        else:
            rows: list[Tensor] = []
            if frame_mask is None:
                frame_mask = encoded.new_ones(encoded.shape[:2])
            if frame_mask.ndim != 2 or frame_mask.shape[0] != encoded.shape[0]:
                raise ValueError("frame_mask must have shape [batch, frames]")
            if frame_mask.shape[1] != encoded.shape[1]:
                frame_mask = F.interpolate(
                    frame_mask.to(dtype=encoded.dtype).unsqueeze(1),
                    size=encoded.shape[1],
                    mode="nearest",
                ).squeeze(1)
            for row, mask in zip(encoded, frame_mask, strict=True):
                valid = row[mask.to(dtype=torch.bool)]
                if valid.shape[0] == 0:
                    raise ValueError("each audio example requires at least one valid frame")
                pooled = F.adaptive_avg_pool1d(
                    valid.transpose(0, 1).unsqueeze(0), self.config.prefix_tokens
                ).transpose(1, 2)
                rows.append(pooled.squeeze(0))
            return self.projector(torch.stack(rows, dim=0))
        return self.projector(pooled)
