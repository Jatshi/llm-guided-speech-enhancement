"""Native audio-prefix projection for an audio-conditioned language model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class AudioConditioningConfig:
    encoder_dim: int
    llm_dim: int
    pooling_stride: int = 4
    freeze_encoder: bool = True

    def __post_init__(self) -> None:
        if self.encoder_dim <= 0 or self.llm_dim <= 0:
            raise ValueError("encoder_dim and llm_dim must be positive")
        if self.pooling_stride <= 0:
            raise ValueError("pooling_stride must be positive")


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

    def forward(self, input_features: Tensor) -> Tensor:
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [batch, frames, features]")
        encoded = self.encoder(input_features).last_hidden_state
        if encoded.ndim != 3 or encoded.shape[-1] != self.config.encoder_dim:
            raise ValueError(
                "audio encoder output must have shape [batch, frames, encoder_dim]"
            )
        pooled = F.avg_pool1d(
            encoded.transpose(1, 2),
            kernel_size=self.config.pooling_stride,
            stride=self.config.pooling_stride,
            ceil_mode=True,
        ).transpose(1, 2)
        return self.projector(pooled)
