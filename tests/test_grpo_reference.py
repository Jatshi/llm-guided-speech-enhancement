from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from lse_v2.training import _attach_grpo_reference


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> FakeParameter:
        self.requires_grad = value
        return self


class FakeReference:
    def __init__(self) -> None:
        self.training = True
        self.parameter = FakeParameter()

    def requires_grad_(self, value: bool) -> FakeReference:
        self.parameter.requires_grad_(value)
        return self

    def eval(self) -> FakeReference:
        self.training = False
        return self


class FakeAccelerator:
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool]] = []

    def prepare_model(self, model: object, *, evaluation_mode: bool) -> object:
        self.calls.append((model, evaluation_mode))
        return ("prepared", model)


def test_attach_grpo_reference_freezes_and_prepares_non_deepspeed_model() -> None:
    reference = FakeReference()
    accelerator = FakeAccelerator()
    trainer = SimpleNamespace(
        ref_model=None,
        is_deepspeed_enabled=False,
        accelerator=accelerator,
    )

    prepared = _attach_grpo_reference(trainer, reference)

    assert prepared == ("prepared", reference)
    assert trainer.ref_model is prepared
    assert reference.parameter.requires_grad is False
    assert reference.training is False
    assert accelerator.calls == [(reference, True)]


def test_attach_grpo_reference_uses_trl_deepspeed_preparation(
    monkeypatch,
) -> None:
    reference = FakeReference()
    prepared = object()
    calls: list[tuple[object, object]] = []
    models_module = ModuleType("trl.models")

    def fake_prepare_deepspeed(model: object, accelerator: object) -> object:
        calls.append((model, accelerator))
        return prepared

    models_module.prepare_deepspeed = fake_prepare_deepspeed
    monkeypatch.setitem(sys.modules, "trl.models", models_module)
    accelerator = object()
    trainer = SimpleNamespace(
        ref_model=None,
        is_deepspeed_enabled=True,
        accelerator=accelerator,
    )

    assert _attach_grpo_reference(trainer, reference) is prepared
    assert trainer.ref_model is prepared
    assert calls == [(reference, accelerator)]
    assert reference.parameter.requires_grad is False
    assert reference.training is False
