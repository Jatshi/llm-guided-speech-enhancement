"""Alignment losses shared by native-audio cDPO and GRPO training."""

from __future__ import annotations


def masked_sequence_log_probability(logits, labels):
    """Return mean next-token log probability over labels not equal to -100."""

    import torch

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must align as [batch, sequence, vocabulary]")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = (
        torch.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    )
    counts = mask.sum(dim=-1).clamp_min(1)
    return (token_logps * mask).sum(dim=-1) / counts


def conservative_dpo_loss(
    policy_chosen,
    policy_rejected,
    reference_chosen,
    reference_rejected,
    *,
    beta: float,
    label_smoothing: float = 0.0,
):
    import torch.nn.functional as functional

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    positive = -functional.logsigmoid(logits)
    negative = -functional.logsigmoid(-logits)
    return ((1 - label_smoothing) * positive + label_smoothing * negative).mean()


def normalized_group_advantages(rewards, epsilon: float = 1e-6):
    """Normalize within prompt groups and preserve zero advantage for ties."""

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [groups, generations>=2]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    spread = rewards.std(dim=1, keepdim=True, unbiased=False)
    normalized = centered / spread.clamp_min(epsilon)
    return normalized.masked_fill(spread <= epsilon, 0.0)
