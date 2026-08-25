from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_audio_conditioned_policy_masks_prefix_and_backpropagates_projector() -> None:
    torch = pytest.importorskip("torch")
    from lse_v2.native_training_pipeline import AudioConditionedPolicy

    class ToyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(11, 6)
            self.head = torch.nn.Linear(6, 11)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, attention_mask, labels, use_cache):
            del attention_mask, use_cache
            logits = self.head(inputs_embeds)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            return SimpleNamespace(logits=logits, loss=loss)

    model = AudioConditionedPolicy(
        ToyLM(), encoder_dim=4, llm_dim=6, prefix_tokens=2, policy_adapter="default"
    )
    output = model(
        encoded_audio=torch.randn(2, 5, 4),
        audio_frame_mask=torch.ones(2, 5),
        input_ids=torch.tensor([[1, 2, 3], [2, 3, 4]]),
        token_attention_mask=torch.ones(2, 3, dtype=torch.long),
        labels=torch.tensor([[-100, 2, 3], [-100, 3, 4]]),
        audio_present=torch.ones(2),
    )
    output.loss.backward()

    assert output.full_labels.shape == (2, 5)
    assert torch.all(output.full_labels[:, :2] == -100)
    assert any(parameter.grad is not None for parameter in model.projector.parameters())


def test_audio_conditioned_policy_casts_cached_frames_to_projector_dtype() -> None:
    torch = pytest.importorskip("torch")
    from lse_v2.native_training_pipeline import AudioConditionedPolicy

    model = AudioConditionedPolicy(
        torch.nn.Linear(6, 6),
        encoder_dim=4,
        llm_dim=6,
        prefix_tokens=2,
        policy_adapter="default",
    )
    model.projector.to(dtype=torch.bfloat16)

    prefix = model._prefix(
        torch.randn(1, 5, 4, dtype=torch.float16),
        torch.ones(1, 5),
        "default",
        torch.ones(1),
    )

    assert prefix.dtype == torch.bfloat16
