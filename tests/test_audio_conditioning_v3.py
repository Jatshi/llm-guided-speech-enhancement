from __future__ import annotations

import pytest


def test_audio_projector_produces_llm_prefix_and_freezes_encoder() -> None:
    torch = pytest.importorskip("torch")
    from lse_v2.audio_conditioning import AudioConditioningConfig, WhisperAudioProjector

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 6)

        def forward(self, input_features):
            class Output:
                pass

            output = Output()
            output.last_hidden_state = self.proj(input_features)
            return output

    module = WhisperAudioProjector(
        Encoder(),
        AudioConditioningConfig(encoder_dim=6, llm_dim=8, pooling_stride=2),
    )
    output = module(torch.ones(2, 10, 4))
    assert output.shape == (2, 5, 8)
    assert all(not parameter.requires_grad for parameter in module.encoder.parameters())
