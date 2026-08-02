from __future__ import annotations

import numpy as np

from lse_v2.closed_loop import (
    ClosedLoopEnhancer,
    EnhancementAction,
    EnhancementPlan,
    MetricSnapshot,
    NumpyDSPExecutor,
)


def test_closed_loop_rolls_back_when_objective_degrades() -> None:
    snapshots = iter(
        [
            MetricSnapshot(si_sdr=4.0, stoi=0.72, wer=0.20),
            MetricSnapshot(si_sdr=2.0, stoi=0.60, wer=0.31),
        ]
    )
    controller = ClosedLoopEnhancer(
        executor=NumpyDSPExecutor(),
        measure=lambda _audio, _sr: next(snapshots),
        min_utility_gain=0.01,
    )
    audio = np.linspace(-0.2, 0.2, 160, dtype=np.float32)
    plan = EnhancementPlan(actions=[EnhancementAction(kind="gain", value=1.5)])

    result = controller.run(audio, sample_rate=16000, plan=plan)

    assert result.decision == "rollback"
    np.testing.assert_allclose(result.audio, audio)
    assert result.before.si_sdr == 4.0
    assert result.after.si_sdr == 2.0


def test_numpy_executor_applies_bounded_gain_without_clipping() -> None:
    executor = NumpyDSPExecutor()
    audio = np.array([-0.8, 0.25, 0.8], dtype=np.float32)
    plan = EnhancementPlan(actions=[EnhancementAction(kind="gain", value=2.0)])

    enhanced = executor.execute(audio, 16000, plan)

    np.testing.assert_allclose(enhanced, np.array([-1.0, 0.5, 1.0], dtype=np.float32))


def test_closed_loop_can_revise_once_then_accept() -> None:
    snapshots = iter(
        [
            MetricSnapshot(si_sdr=2.0, stoi=0.60),
            MetricSnapshot(si_sdr=1.0, stoi=0.55),
            MetricSnapshot(si_sdr=5.0, stoi=0.75),
        ]
    )
    controller = ClosedLoopEnhancer(
        executor=NumpyDSPExecutor(),
        measure=lambda _audio, _sr: next(snapshots),
        reviser=lambda _plan, _before, _after: EnhancementPlan(
            actions=[EnhancementAction(kind="dc_remove")]
        ),
        max_revisions=1,
        min_utility_gain=0.01,
    )
    audio = np.linspace(-0.2, 0.2, 160, dtype=np.float32)

    result = controller.run(
        audio,
        sample_rate=16000,
        plan=EnhancementPlan(actions=[EnhancementAction(kind="gain", value=2.0)]),
    )

    assert result.decision == "accept"
    assert len(result.attempts) == 2
    assert result.attempts[0].decision == "revise"
    assert result.attempts[1].decision == "accept"
