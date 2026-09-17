"""Compact three-stream MM-DiT for prescription-conditioned enhancement."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class MMDiTConfig:
    in_channels: int = 2
    model_dim: int = 384
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    patch_frequency: int = 16
    patch_time: int = 8
    prescription_tokens: int = 8
    prescription_vocab: int = 1024
    field_vocab: int = 64
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.depth < 1 or self.patch_frequency < 1 or self.patch_time < 1:
            raise ValueError("depth and patch sizes must be positive")


def _sinusoidal(values: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scales = torch.exp(
        -math.log(10_000) * torch.arange(half, device=values.device, dtype=values.dtype) / half
    )
    angles = values.unsqueeze(-1) * scales
    embedded = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(embedded, (0, dim - embedded.shape[-1]))


def _position(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=dtype)
    return _sinusoidal(positions, dim).unsqueeze(0)


class PrescriptionEncoder(nn.Module):
    def __init__(self, config: MMDiTConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.field = nn.Embedding(config.field_vocab, dim, padding_idx=0)
        self.category = nn.Embedding(config.prescription_vocab, dim, padding_idx=0)
        self.value = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, fields: torch.Tensor, categories: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = fields.ne(0)
        encoded = self.field(fields) + self.category(categories) + self.value(values.unsqueeze(-1))
        encoded = encoded + _position(
            encoded.shape[1], encoded.shape[2], encoded.device, encoded.dtype
        )
        return self.norm(encoded) * mask.unsqueeze(-1), mask


class JointAttentionBlock(nn.Module):
    def __init__(self, config: MMDiTConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.heads = config.heads
        self.head_dim = dim // config.heads
        self.target_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.observed_norm = nn.LayerNorm(dim)
        self.prescription_norm = nn.LayerNorm(dim)
        self.target_qkv = nn.Linear(dim, dim * 3)
        self.observed_qkv = nn.Linear(dim, dim * 3)
        self.prescription_qkv = nn.Linear(dim, dim * 3)
        self.target_out = nn.Linear(dim, dim)
        self.observed_out = nn.Linear(dim, dim)
        self.prescription_out = nn.Linear(dim, dim)
        hidden = int(dim * config.mlp_ratio)
        self.target_mlp = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False),
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, dim),
        )
        self.observed_mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.prescription_mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def _qkv(self, layer: nn.Linear, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, tokens, _ = value.shape
        qkv = layer(value).view(batch, tokens, 3, self.heads, self.head_dim)
        return tuple(item.transpose(1, 2) for item in qkv.unbind(dim=2))

    def forward(
        self,
        target: torch.Tensor,
        observed: torch.Tensor,
        prescription: torch.Tensor,
        time_embedding: torch.Tensor,
        prescription_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(time_embedding).chunk(
            6, dim=-1
        )
        target_norm = self.target_norm(target) * (1 + scale_a[:, None]) + shift_a[:, None]
        tq, tk, tv = self._qkv(self.target_qkv, target_norm)
        oq, ok, ov = self._qkv(self.observed_qkv, self.observed_norm(observed))
        pq, pk, pv = self._qkv(self.prescription_qkv, self.prescription_norm(prescription))
        query = torch.cat((tq, oq, pq), dim=2)
        key = torch.cat((tk, ok, pk), dim=2)
        value = torch.cat((tv, ov, pv), dim=2)
        audio_tokens = target.shape[1] + observed.shape[1]
        key_allowed = torch.cat(
            (
                torch.ones(target.shape[0], audio_tokens, device=target.device, dtype=torch.bool),
                prescription_mask,
            ),
            dim=1,
        )
        attention_bias = torch.zeros(
            target.shape[0], 1, 1, key_allowed.shape[1], device=target.device, dtype=target.dtype
        )
        attention_bias.masked_fill_(~key_allowed[:, None, None, :], torch.finfo(target.dtype).min)
        joined = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias, dropout_p=0.0
        )
        joined = joined.transpose(1, 2).reshape(target.shape[0], -1, target.shape[2])
        target_delta, observed_delta, prescription_delta = joined.split(
            (target.shape[1], observed.shape[1], prescription.shape[1]), dim=1
        )
        target = target + gate_a[:, None] * self.target_out(target_delta)
        observed = observed + self.observed_out(observed_delta)
        prescription = prescription + self.prescription_out(prescription_delta)
        target_norm = self.target_mlp[0](target) * (1 + scale_m[:, None]) + shift_m[:, None]
        target = target + gate_m[:, None] * self.target_mlp[1:](target_norm)
        observed = observed + self.observed_mlp(observed)
        prescription = prescription + self.prescription_mlp(prescription)
        prescription = prescription * prescription_mask.unsqueeze(-1)
        return target, observed, prescription


class PrescriptionConditionedMMDiT(nn.Module):
    def __init__(self, config: MMDiTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        patch = (config.patch_frequency, config.patch_time)
        self.target_embed = nn.Conv2d(config.in_channels, config.model_dim, patch, stride=patch)
        self.observed_embed = nn.Conv2d(config.in_channels, config.model_dim, patch, stride=patch)
        self.prescription_encoder = PrescriptionEncoder(config)
        self.target_modality = nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)
        self.observed_modality = nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model_dim * 4, config.model_dim),
        )
        self.blocks = nn.ModuleList(JointAttentionBlock(config) for _ in range(config.depth))
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.output = nn.ConvTranspose2d(config.model_dim, config.in_channels, patch, stride=patch)

    def export_config(self) -> dict:
        return asdict(self.config)

    def _patch(self, value: torch.Tensor, layer: nn.Conv2d) -> tuple[torch.Tensor, tuple[int, int]]:
        frequency, frames = value.shape[-2:]
        pad_f = (-frequency) % self.config.patch_frequency
        pad_t = (-frames) % self.config.patch_time
        value = F.pad(value, (0, pad_t, 0, pad_f))
        embedded = layer(value)
        tokens = embedded.flatten(2).transpose(1, 2)
        tokens = tokens + _position(tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
        return tokens, (pad_f, pad_t)

    def forward(
        self,
        target: torch.Tensor,
        observed: torch.Tensor,
        fields: torch.Tensor,
        categories: torch.Tensor,
        values: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if target.shape != observed.shape:
            raise ValueError("target and observed latent shapes must match")
        if target.ndim != 4 or target.shape[1] != self.config.in_channels:
            raise ValueError("latents must have shape [batch, channels, frequency, time]")
        target_tokens, padding = self._patch(target, self.target_embed)
        observed_tokens, observed_padding = self._patch(observed, self.observed_embed)
        if padding != observed_padding:
            raise AssertionError("target and observed padding unexpectedly differ")
        target_tokens = target_tokens + self.target_modality
        observed_tokens = observed_tokens + self.observed_modality
        prescription, prescription_mask = self.prescription_encoder(fields, categories, values)
        time_embedding = self.time_mlp(
            _sinusoidal(timestep.to(target.dtype), self.config.model_dim)
        )
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                target_tokens, observed_tokens, prescription = checkpoint(
                    block,
                    target_tokens,
                    observed_tokens,
                    prescription,
                    time_embedding,
                    prescription_mask,
                    use_reentrant=False,
                )
            else:
                target_tokens, observed_tokens, prescription = block(
                    target_tokens,
                    observed_tokens,
                    prescription,
                    time_embedding,
                    prescription_mask,
                )
        pad_f, pad_t = padding
        frequency_patches = (target.shape[-2] + pad_f) // self.config.patch_frequency
        time_patches = (target.shape[-1] + pad_t) // self.config.patch_time
        feature_map = (
            self.output_norm(target_tokens)
            .transpose(1, 2)
            .reshape(target.shape[0], self.config.model_dim, frequency_patches, time_patches)
        )
        output = self.output(feature_map)
        return output[..., : target.shape[-2], : target.shape[-1]]
