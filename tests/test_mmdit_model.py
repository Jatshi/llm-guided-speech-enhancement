import torch
from torch import nn

from lse_v2.mmdit.codec import ComplexSTFTCodec, STFTCodecConfig
from lse_v2.mmdit.flow import RectifiedFlow, RectifiedFlowConfig
from lse_v2.mmdit.model import MMDiTConfig, PrescriptionConditionedMMDiT
from lse_v2.mmdit.safety import EnhancementGate, EnhancementGateConfig


def _model() -> PrescriptionConditionedMMDiT:
    return PrescriptionConditionedMMDiT(
        MMDiTConfig(
            in_channels=2,
            model_dim=32,
            depth=2,
            heads=4,
            patch_frequency=4,
            patch_time=4,
            prescription_tokens=8,
            prescription_vocab=128,
            dropout=0.0,
        )
    )


def test_mmdit_preserves_latent_shape_and_uses_prescription():
    torch.manual_seed(7)
    model = _model().eval()
    target = torch.randn(2, 2, 16, 20)
    observed = torch.randn_like(target)
    timestep = torch.tensor([0.2, 0.8])
    fields = torch.ones(2, 8, dtype=torch.long)
    categories_a = torch.ones(2, 8, dtype=torch.long)
    categories_b = categories_a.clone()
    categories_b[:, 0] = 17
    values = torch.zeros(2, 8)
    with torch.no_grad():
        first = model(target, observed, fields, categories_a, values, timestep)
        second = model(target, observed, fields, categories_b, values, timestep)
    assert first.shape == target.shape
    assert not torch.allclose(first, second)


def test_rectified_flow_loss_and_sampler_are_finite():
    torch.manual_seed(3)
    model = _model()
    flow = RectifiedFlow(RectifiedFlowConfig(steps=3, prescription_dropout=0.0))
    clean = torch.randn(2, 2, 16, 20)
    observed = torch.randn_like(clean)
    fields = torch.ones(2, 8, dtype=torch.long)
    categories = torch.ones(2, 8, dtype=torch.long)
    values = torch.zeros(2, 8)
    loss = flow.training_loss(model, clean, observed, fields, categories, values)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    with torch.no_grad():
        generated = flow.sample(
            model.eval(), observed[:1], fields[:1], categories[:1], values[:1], seed=11
        )
    assert generated.shape == observed[:1].shape
    assert torch.isfinite(generated).all()


class _ZeroVelocity(nn.Module):
    def forward(self, current, observed, fields, categories, values, timestep):
        del observed, fields, categories, values, timestep
        return torch.zeros_like(current)


def test_observed_source_sampler_starts_from_noisy_latent():
    observed = torch.randn(1, 2, 8, 10)
    fields = torch.zeros(1, 8, dtype=torch.long)
    flow = RectifiedFlow(
        RectifiedFlowConfig(steps=4, source_mode="observed", prescription_dropout=0.0)
    )
    generated = flow.sample(_ZeroVelocity(), observed, fields, fields, torch.zeros(1, 8), seed=19)
    assert torch.equal(generated, observed)


def test_observed_source_identity_target_has_zero_flow_loss():
    observed = torch.randn(2, 2, 8, 10)
    fields = torch.ones(2, 8, dtype=torch.long)
    flow = RectifiedFlow(
        RectifiedFlowConfig(steps=2, source_mode="observed", prescription_dropout=0.0)
    )
    loss = flow.training_loss(
        _ZeroVelocity(), observed, observed, fields, fields, torch.zeros(2, 8)
    )
    assert loss.total < 1e-7
    assert loss.velocity_mse == 0


def test_observed_source_reports_mismatch_noop_loss():
    torch.manual_seed(5)
    model = _model()
    observed = torch.randn(2, 2, 8, 8)
    clean = torch.randn_like(observed)
    fields = torch.ones(2, 8, dtype=torch.long)
    categories = torch.stack((torch.ones(8, dtype=torch.long), torch.full((8,), 2)))
    flow = RectifiedFlow(
        RectifiedFlowConfig(
            steps=2,
            source_mode="observed",
            prescription_dropout=0.0,
            mismatch_noop_weight=0.5,
        )
    )
    loss = flow.training_loss(model, clean, observed, fields, categories, torch.zeros(2, 8))
    assert torch.isfinite(loss.mismatch_noop_mse)
    assert loss.mismatch_noop_mse > 0


def test_complex_stft_codec_round_trip():
    codec = ComplexSTFTCodec(STFTCodecConfig(n_fft=128, hop_length=32, scale=16.0))
    unscaled = ComplexSTFTCodec(STFTCodecConfig(n_fft=128, hop_length=32, scale=1.0))
    waveform = torch.sin(torch.linspace(0, 30, 1600)).unsqueeze(0)
    latent = codec.encode(waveform)
    restored = codec.decode(latent, length=waveform.shape[-1])
    assert latent.shape[1] == 2
    assert torch.allclose(latent, unscaled.encode(waveform) * 16.0)
    assert restored.shape == waveform.shape
    assert torch.mean((restored - waveform) ** 2) < 1e-8


def test_quality_gate_falls_back_on_clipping_and_accepts_safe_output():
    gate = EnhancementGate(EnhancementGateConfig(max_peak=1.0, max_energy_ratio=4.0))
    noisy = torch.full((1, 100), 0.1)
    unsafe = torch.full((1, 100), 2.0)
    selected, unsafe_report = gate.select(noisy, unsafe, planner_confidence=0.9)
    assert unsafe_report["used_fallback"] is True
    assert torch.equal(selected, noisy)
    safe = noisy * 1.2
    selected, safe_report = gate.select(noisy, safe, planner_confidence=0.9)
    assert safe_report["used_fallback"] is False
    assert torch.equal(selected, safe)


def test_gradient_checkpointing_path_backpropagates():
    config = _model().config
    model = PrescriptionConditionedMMDiT(
        MMDiTConfig(**{**model_config_dict(config), "gradient_checkpointing": True})
    ).train()
    target = torch.randn(1, 2, 8, 8, requires_grad=True)
    fields = torch.ones(1, 8, dtype=torch.long)
    output = model(
        target,
        torch.randn_like(target),
        fields,
        fields,
        torch.zeros(1, 8),
        torch.tensor([0.5]),
    )
    output.mean().backward()
    assert target.grad is not None


def model_config_dict(config: MMDiTConfig) -> dict:
    from dataclasses import asdict

    return asdict(config)
