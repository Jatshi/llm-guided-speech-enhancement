from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from lse_v2.mmdit.adaptive_planner import (  # noqa: E402
    AdaptivePlanningConfig,
    choose_planning_route,
    planning_benefit_label,
)
from lse_v2.mmdit.config import load_mmdit_config  # noqa: E402
from lse_v2.mmdit.contracts import (  # noqa: E402
    enhance_script_tokens,
    validate_enhance_script,
)
from lse_v2.mmdit.curriculum import CurriculumSchedule  # noqa: E402
from lse_v2.mmdit.enhance_script import bootstrap_enhance_script  # noqa: E402
from lse_v2.mmdit.enhancement_reward import (  # noqa: E402
    score_enhancement_outcome,
    select_preference_pair,
)
from lse_v2.mmdit.evaluate import _degradation_summaries, _summaries  # noqa: E402
from lse_v2.mmdit.flow import RectifiedFlow, RectifiedFlowConfig  # noqa: E402
from lse_v2.mmdit.losses import (  # noqa: E402
    MultiResolutionSTFTLoss,
    semantic_consistency_loss,
)
from lse_v2.mmdit.model import MMDiTConfig, PrescriptionConditionedMMDiT  # noqa: E402
from lse_v2.mmdit.planner_preference_data import build_preference_rows  # noqa: E402
from lse_v2.mmdit.train import _phase_scoped_best, _prepare_metrics_log  # noqa: E402


def test_identity_initialization_predicts_exact_zero_velocity() -> None:
    model = PrescriptionConditionedMMDiT(
        MMDiTConfig(
            model_dim=32,
            depth=2,
            heads=4,
            patch_frequency=4,
            patch_time=4,
            prescription_vocab=128,
            identity_init=True,
        )
    ).eval()
    current = torch.randn(2, 2, 12, 16)
    observed = torch.randn_like(current)
    fields = torch.ones(2, 8, dtype=torch.long)
    categories = torch.randint(1, 16, (2, 8))
    values = torch.randn(2, 8)

    with torch.no_grad():
        velocity = model(
            current,
            observed,
            fields,
            categories,
            values,
            torch.tensor([0.25, 0.75]),
        )

    assert torch.count_nonzero(velocity) == 0
    for block in model.blocks:
        assert torch.count_nonzero(block.modulation[-1].weight) == 0
        assert torch.count_nonzero(block.modulation[-1].bias) == 0


def test_multi_resolution_stft_loss_is_zero_for_identity_and_has_gradient() -> None:
    objective = MultiResolutionSTFTLoss(fft_sizes=(64, 128), hop_sizes=(16, 32))
    reference = torch.randn(2, 512)
    same = objective(reference, reference)
    assert same < 1e-7

    estimate = (reference + 0.05 * torch.randn_like(reference)).requires_grad_()
    changed = objective(estimate, reference)
    assert changed > same
    changed.backward()
    assert estimate.grad is not None
    assert torch.isfinite(estimate.grad).all()


class _TinySemanticTeacher(nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                waveform[:, ::4],
                waveform[:, 1::4],
                waveform[:, 2::4],
                waveform[:, 3::4],
            ),
            dim=-1,
        )


def test_semantic_consistency_loss_preserves_input_gradient_only() -> None:
    teacher = _TinySemanticTeacher()
    clean = torch.randn(2, 400)
    enhanced = (clean + 0.1 * torch.randn_like(clean)).requires_grad_()
    loss = semantic_consistency_loss(teacher, enhanced, clean)
    loss.backward()
    assert 0 <= loss <= 2
    assert enhanced.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_curriculum_selects_contiguous_phases_and_overrides_flow() -> None:
    schedule = CurriculumSchedule.from_config(
        [
            {
                "name": "oracle_alignment",
                "until_step": 100,
                "flow": {"prescription_dropout": 0.0, "mismatch_noop_weight": 0.0},
                "loss_weights": {"mrstft": 0.05, "semantic": 0.0},
                "lr_scale": 1.0,
            },
            {
                "name": "safety_alignment",
                "until_step": 200,
                "flow": {"prescription_dropout": 0.15, "mismatch_noop_weight": 0.5},
                "loss_weights": {"mrstft": 0.1, "semantic": 0.05},
                "lr_scale": 0.5,
            },
        ],
        max_steps=200,
    )

    assert schedule.phase_for_step(1).name == "oracle_alignment"
    assert schedule.phase_for_step(100).name == "oracle_alignment"
    second = schedule.phase_for_step(101)
    assert second.name == "safety_alignment"
    assert second.flow_overrides["mismatch_noop_weight"] == 0.5
    assert second.semantic_weight == 0.05
    assert schedule.phase_for_step(200).lr_scale == 0.5


def _script() -> dict:
    return {
        "schema_version": "lse.enhance_script.v1",
        "duration_ms": 2000,
        "preserve": {
            "speech_content": True,
            "speaker_identity": True,
            "prosody": True,
        },
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 640,
                "diagnosis": "pink",
                "action": "spectral_subtraction",
                "strength": 0.35,
                "confidence": 0.92,
            },
            {
                "start_ms": 640,
                "end_ms": 2000,
                "diagnosis": "clean",
                "action": "hold",
                "strength": 0.0,
                "confidence": 0.88,
            },
        ],
    }


def test_enhance_script_is_validated_and_tokenized_deterministically() -> None:
    script = _script()
    validate_enhance_script(script)
    first = enhance_script_tokens(script, max_tokens=16)
    second = enhance_script_tokens(script, max_tokens=16)
    assert first == second
    assert len(first) == 16
    assert any(field >= 20 for field, _, _ in first)


def test_enhance_script_rejects_overlaps() -> None:
    script = _script()
    script["segments"][1]["start_ms"] = 600
    with pytest.raises(ValueError, match="overlap"):
        validate_enhance_script(script)


@pytest.mark.parametrize(
    ("label", "confidence", "expected"),
    [
        ("pink", 0.93, "direct_template"),
        ("pink", 0.55, "llm_planner"),
        ("clean", 0.95, "abstain"),
        ("pink", 0.12, "abstain"),
    ],
)
def test_adaptive_planning_route(label: str, confidence: float, expected: str) -> None:
    decision = choose_planning_route(
        label,
        confidence,
        AdaptivePlanningConfig(
            direct_confidence=0.85,
            minimum_action_confidence=0.35,
            supported_diagnoses=("white", "pink"),
        ),
    )
    assert decision.route == expected


def test_counterfactual_planning_label_requires_a_real_gain() -> None:
    assert planning_benefit_label(0.2, 0.31, margin=0.05) is True
    assert planning_benefit_label(0.2, 0.23, margin=0.05) is False


class _ZeroVelocity(nn.Module):
    def forward(self, current, observed, fields, categories, values, timestep):
        del observed, fields, categories, values, timestep
        return torch.zeros_like(current)


def test_flow_loss_exposes_differentiable_x0_estimate() -> None:
    clean = torch.randn(2, 2, 8, 10)
    observed = torch.randn_like(clean)
    fields = torch.ones(2, 8, dtype=torch.long)
    loss = RectifiedFlow(
        RectifiedFlowConfig(source_mode="observed", prescription_dropout=0.0)
    ).training_loss(_ZeroVelocity(), clean, observed, fields, fields, torch.zeros(2, 8))
    assert loss.estimated_x0.shape == clean.shape
    assert torch.isfinite(loss.estimated_x0).all()


def test_global_prescription_bootstraps_a_safe_time_local_script() -> None:
    prescription = {
        "diagnosis": {"noise_type": "pink", "reverb": False},
        "actions": [{"type": "spectral_subtraction", "reduction_db": 6.0}],
        "confidence": 0.8,
    }
    script = bootstrap_enhance_script(prescription, duration_ms=2000)
    validate_enhance_script(script)
    assert script["segments"][0]["action"] == "spectral_subtraction"
    assert script["segments"][0]["strength"] == pytest.approx(0.25)


def test_v6_autodl_config_is_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_mmdit_config(root / "configs" / "mmdit_v6_stepaudio3_32gb.json")
    assert config["model"]["identity_init"] is True
    assert config["model"]["prescription_tokens"] == 16
    assert config["training"]["curriculum"][-1]["until_step"] == 12000
    assert config["training"]["auxiliary_losses"]["semantic"]["model_name"].startswith("microsoft/")


def test_v6_smoke_config_exercises_identity_curriculum_and_mrstft() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_mmdit_config(root / "configs" / "mmdit_v6_stepaudio3_smoke.json")
    assert config["model"]["identity_init"] is True
    assert config["model"]["prescription_tokens"] == 16
    assert len(config["training"]["curriculum"]) == 2
    assert config["training"]["curriculum"][0]["loss_weights"]["mrstft"] > 0


def test_v6_canary_exercises_the_real_model_and_semantic_teacher() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_mmdit_config(root / "configs" / "mmdit_v6_stepaudio3_canary.json")
    assert config["model"]["model_dim"] == 384
    assert config["training"]["max_steps"] == 20
    assert config["training"]["curriculum"][0]["loss_weights"]["semantic"] > 0
    assert config["evaluation"]["max_samples"] == 4


def test_best_checkpoint_resets_when_curriculum_objective_changes() -> None:
    best, phase, improved = _phase_scoped_best(
        "oracle_alignment",
        0.19,
        best_phase="oracle_alignment",
        best_validation=0.18,
    )
    assert (best, phase, improved) == (0.18, "oracle_alignment", False)

    best, phase, improved = _phase_scoped_best(
        "safety_alignment",
        0.24,
        best_phase=phase,
        best_validation=best,
    )
    assert (best, phase, improved) == (0.24, "safety_alignment", True)


def test_old_checkpoint_without_best_phase_establishes_fresh_best() -> None:
    best, phase, improved = _phase_scoped_best(
        "safety_alignment",
        0.23,
        best_phase=None,
        best_validation=0.18,
    )
    assert (best, phase, improved) == (0.23, "safety_alignment", True)


def test_evaluation_summaries_keep_clean_safety_slice_separate() -> None:
    rows = [
        {
            "variant": "model",
            "degradation_type": "clean",
            "si_sdr_improvement": -100.0,
            "used_fallback": True,
        },
        {
            "variant": "model",
            "degradation_type": "white",
            "si_sdr_improvement": 1.0,
            "used_fallback": False,
        },
        {
            "variant": "model",
            "degradation_type": "pink",
            "si_sdr_improvement": 3.0,
            "used_fallback": False,
        },
    ]

    overall = _summaries(rows)[0]
    grouped = _degradation_summaries(rows)

    assert overall["si_sdr_improvement_mean"] == -32.0
    assert overall["si_sdr_improvement_median"] == 1.0
    assert grouped["clean"][0]["records"] == 1
    assert grouped["corrupted_only"][0]["records"] == 2
    assert grouped["corrupted_only"][0]["si_sdr_improvement_mean"] == 2.0


def test_resume_truncates_and_deduplicates_metrics_log(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"step": 1, "loss": 1.0}',
                '{"step": 2, "loss": 0.9}',
                '{"step": 2, "loss": 0.8}',
                '{"step": 3, "loss": 0.7}',
                "partial-json",
            ]
        ),
        encoding="utf-8",
    )

    _prepare_metrics_log(path, resume_step=2)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.8}]


def test_enhancement_reward_prefers_quality_without_content_damage() -> None:
    safe = score_enhancement_outcome(
        {
            "si_sdr_improvement": 1.2,
            "snr_improvement": 0.8,
            "pesq_improvement": 0.15,
            "stoi_improvement": 0.02,
            "cer_before": 0.12,
            "cer_after": 0.10,
            "speaker_similarity": 0.94,
            "artifact_penalty": 0.0,
            "waveform_safe": True,
        }
    )
    content_damaged = score_enhancement_outcome(
        {
            "si_sdr_improvement": 2.0,
            "snr_improvement": 1.2,
            "pesq_improvement": 0.2,
            "stoi_improvement": 0.03,
            "cer_before": 0.12,
            "cer_after": 0.65,
            "speaker_similarity": 0.7,
            "artifact_penalty": 0.1,
            "waveform_safe": True,
        }
    )
    assert safe > content_damaged
    assert content_damaged == -1.0


def test_preference_pair_uses_largest_verified_reward_gap() -> None:
    candidates = [
        {"id": "middle", "reward": 0.2},
        {"id": "best", "reward": 0.8},
        {"id": "worst", "reward": -0.4},
    ]
    pair = select_preference_pair(candidates, min_gap=0.3)
    assert pair is not None
    assert pair["chosen"]["id"] == "best"
    assert pair["rejected"]["id"] == "worst"
    assert pair["reward_gap"] == pytest.approx(1.2)


def test_preference_builder_uses_executed_outcomes_not_text_scores() -> None:
    common = {
        "sample_id": "utt-001",
        "prompt": "diagnose and enhance",
        "metrics": {
            "snr_improvement": 1.0,
            "pesq_improvement": 0.1,
            "stoi_improvement": 0.02,
            "cer_before": 0.1,
            "speaker_similarity": 0.95,
            "artifact_penalty": 0.0,
            "waveform_safe": True,
        },
    }
    safe = {
        **common,
        "candidate_id": "safe",
        "prescription": {"actions": [{"type": "hold"}]},
        "metrics": {
            **common["metrics"],
            "si_sdr_improvement": 1.5,
            "cer_after": 0.09,
        },
    }
    damaged = {
        **common,
        "candidate_id": "aggressive",
        "prescription": {"actions": [{"type": "overclean"}]},
        "metrics": {
            **common["metrics"],
            "si_sdr_improvement": 3.0,
            "cer_after": 0.8,
        },
    }
    pairs, report = build_preference_rows([damaged, safe], min_gap=0.1)
    assert report["preference_pairs"] == 1
    assert pairs[0]["chosen_candidate_id"] == "safe"
    assert pairs[0]["rejected_candidate_id"] == "aggressive"
    assert pairs[0]["reward_source"] == "executed_enhancement_metrics"
