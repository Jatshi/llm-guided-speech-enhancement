from __future__ import annotations

import pytest


def test_conservative_dpo_has_finite_nonzero_gradient() -> None:
    torch = pytest.importorskip("torch")
    from lse_v2.native_alignment import conservative_dpo_loss

    policy_chosen = torch.tensor([1.2, 0.4], requires_grad=True)
    policy_rejected = torch.tensor([0.2, 0.3], requires_grad=True)
    reference_chosen = torch.tensor([0.8, 0.2])
    reference_rejected = torch.tensor([0.1, 0.1])

    loss = conservative_dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
        label_smoothing=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert policy_chosen.grad is not None
    assert torch.count_nonzero(policy_chosen.grad) > 0


def test_group_advantages_are_zero_for_saturated_group() -> None:
    torch = pytest.importorskip("torch")
    from lse_v2.native_alignment import normalized_group_advantages

    saturated = normalized_group_advantages(torch.tensor([[0.9, 0.9], [0.2, 0.2]]))
    varied = normalized_group_advantages(torch.tensor([[0.2, 0.8], [0.4, 0.4]]))

    torch.testing.assert_close(saturated, torch.zeros_like(saturated))
    assert varied[0, 0] < 0 < varied[0, 1]
    assert torch.all(varied[1] == 0)


def test_masked_sequence_log_probability_ignores_prompt_tokens() -> None:
    torch = pytest.importorskip("torch")
    from lse_v2.native_alignment import masked_sequence_log_probability

    logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]], dtype=torch.float32)
    labels = torch.tensor([[-100, -100, 0, 1]])

    score = masked_sequence_log_probability(logits, labels)

    assert score.shape == (1,)
    assert score.item() > -0.1
