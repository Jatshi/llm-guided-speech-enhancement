"""Rectified-flow objective and deterministic Euler sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RectifiedFlowConfig:
    steps: int = 8
    prescription_dropout: float = 0.1
    x0_l1_weight: float = 0.1
    guidance_scale: float = 1.0
    source_mode: str = "gaussian"
    mismatch_noop_weight: float = 0.0


@dataclass
class FlowLoss:
    total: torch.Tensor
    velocity_mse: torch.Tensor
    x0_l1: torch.Tensor
    mismatch_noop_mse: torch.Tensor
    estimated_x0: torch.Tensor


class RectifiedFlow:
    def __init__(self, config: RectifiedFlowConfig | None = None) -> None:
        self.config = config or RectifiedFlowConfig()
        if self.config.steps < 1:
            raise ValueError("steps must be positive")
        if not 0 <= self.config.prescription_dropout < 1:
            raise ValueError("prescription_dropout must be in [0, 1)")
        if self.config.source_mode not in {"gaussian", "observed"}:
            raise ValueError("source_mode must be gaussian or observed")
        if self.config.mismatch_noop_weight < 0:
            raise ValueError("mismatch_noop_weight must be non-negative")
        if self.config.source_mode != "observed" and self.config.mismatch_noop_weight:
            raise ValueError("mismatch_noop_weight requires source_mode=observed")

    def training_loss(
        self,
        model,
        clean: torch.Tensor,
        observed: torch.Tensor,
        fields: torch.Tensor,
        categories: torch.Tensor,
        values: torch.Tensor,
    ) -> FlowLoss:
        batch = clean.shape[0]
        timestep = torch.rand(batch, device=clean.device, dtype=clean.dtype)
        source = observed if self.config.source_mode == "observed" else torch.randn_like(clean)
        target = clean
        original_fields = fields
        original_categories = categories
        original_values = values
        drop = torch.zeros(batch, device=clean.device, dtype=torch.bool)
        if self.config.prescription_dropout:
            drop = torch.rand(batch, device=clean.device) < self.config.prescription_dropout
            fields = fields.clone()
            categories = categories.clone()
            values = values.clone()
            fields[drop] = 0
            categories[drop] = 0
            values[drop] = 0
            if self.config.source_mode == "observed":
                # An absent prescription is a safety abstention, not permission to
                # perform an unconstrained enhancement.
                target = torch.where(drop.view(batch, 1, 1, 1), observed, clean)
        expanded_t = timestep.view(batch, 1, 1, 1)
        current = (1 - expanded_t) * target + expanded_t * source
        velocity = source - target
        predicted = model(current, observed, fields, categories, values, timestep)
        velocity_mse = F.mse_loss(predicted, velocity)
        estimated_x0 = current - expanded_t * predicted
        x0_l1 = F.l1_loss(estimated_x0, target)

        mismatch_noop_mse = predicted.new_zeros(())
        if self.config.mismatch_noop_weight and batch > 1:
            mismatch_fields = original_fields.roll(1, dims=0)
            mismatch_categories = original_categories.roll(1, dims=0)
            mismatch_values = original_values.roll(1, dims=0)
            differs = (
                (mismatch_fields != original_fields).any(dim=1)
                | (mismatch_categories != original_categories).any(dim=1)
                | ~torch.isclose(mismatch_values, original_values).all(dim=1)
            )
            if differs.any():
                mismatch_t = torch.rand(batch, device=clean.device, dtype=clean.dtype)
                mismatch_prediction = model(
                    observed,
                    observed,
                    mismatch_fields,
                    mismatch_categories,
                    mismatch_values,
                    mismatch_t,
                )
                per_item = mismatch_prediction.square().flatten(1).mean(dim=1)
                mismatch_noop_mse = per_item[differs].mean()

        total = (
            velocity_mse
            + self.config.x0_l1_weight * x0_l1
            + self.config.mismatch_noop_weight * mismatch_noop_mse
        )
        return FlowLoss(
            total=total,
            velocity_mse=velocity_mse,
            x0_l1=x0_l1,
            mismatch_noop_mse=mismatch_noop_mse,
            estimated_x0=estimated_x0,
        )

    def sample(
        self,
        model,
        observed: torch.Tensor,
        fields: torch.Tensor,
        categories: torch.Tensor,
        values: torch.Tensor,
        *,
        seed: int = 0,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        count = steps or self.config.steps
        if count < 1:
            raise ValueError("sampling steps must be positive")
        scale = self.config.guidance_scale if guidance_scale is None else guidance_scale
        if self.config.source_mode == "observed":
            current = observed.clone()
        else:
            generator = torch.Generator(device=observed.device).manual_seed(seed)
            current = torch.randn(
                observed.shape, generator=generator, device=observed.device, dtype=observed.dtype
            )
        delta = 1.0 / count
        for index in range(count):
            t_value = 1.0 - index * delta
            timestep = torch.full(
                (observed.shape[0],), t_value, device=observed.device, dtype=observed.dtype
            )
            conditional = model(current, observed, fields, categories, values, timestep)
            if scale != 1.0:
                zeros_i = torch.zeros_like(fields)
                zeros_f = torch.zeros_like(values)
                unconditional = model(current, observed, zeros_i, zeros_i, zeros_f, timestep)
                conditional = unconditional + scale * (conditional - unconditional)
            current = current - delta * conditional
        return current
