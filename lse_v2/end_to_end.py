"""End-to-end prescription execution, objective verification, and audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .dsp import ProductionDSPExecutor, plan_from_prescription
from .io import utc_now, write_json_atomic
from .metrics import MetricComparison, ObjectiveMetricSuite


@dataclass(frozen=True, slots=True)
class EnhancementRequest:
    audio_path: Path
    prescription: str
    output_dir: Path
    clean_reference: Path | None = None
    target_sample_rate: int = 16_000
    minimum_si_sdr_gain_db: float = 0.0
    allow_unverified: bool = False


@dataclass(frozen=True, slots=True)
class EnhancementRunResult:
    decision: Literal["accept", "rollback", "unverified"]
    output_audio: Path
    candidate_audio: Path
    audit_report: Path
    metrics: MetricComparison


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    import soundfile as sf  # type: ignore[import-untyped]
    from scipy.signal import resample_poly

    waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError(f"invalid audio waveform: {path}")
    if source_rate != sample_rate:
        divisor = int(np.gcd(source_rate, sample_rate))
        waveform = resample_poly(waveform, sample_rate // divisor, source_rate // divisor)
    return np.asarray(waveform, dtype=np.float32)


def _write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(waveform, dtype=np.float32), sample_rate, subtype="PCM_16")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_enhancement(
    request: EnhancementRequest,
    *,
    metric_suite: ObjectiveMetricSuite | None = None,
) -> EnhancementRunResult:
    source_path = request.audio_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _load_audio(source_path, request.target_sample_rate)
    clean = (
        _load_audio(request.clean_reference.expanduser().resolve(), request.target_sample_rate)
        if request.clean_reference is not None
        else None
    )
    plan = plan_from_prescription(request.prescription)
    candidate = ProductionDSPExecutor().execute(source, request.target_sample_rate, plan)
    suite = metric_suite or ObjectiveMetricSuite.with_installed_backends()
    comparison = suite.compare(
        source,
        candidate,
        request.target_sample_rate,
        clean_reference=clean,
    )
    if clean is not None:
        shortest_seconds = min(source.size, candidate.size, clean.size) / request.target_sample_rate
        frame_seconds = min(0.5, shortest_seconds)
        frame_metrics = (
            suite.compare_frames(
                source,
                candidate,
                request.target_sample_rate,
                clean_reference=clean,
                frame_seconds=frame_seconds,
            )
            if frame_seconds * request.target_sample_rate >= 2
            else {"available": False, "reason": "audio_too_short"}
        )
    else:
        frame_metrics = {"available": False, "reason": "clean_reference_required"}
    if "si_sdr" in comparison.deltas:
        decision: Literal["accept", "rollback", "unverified"] = (
            "accept"
            if comparison.deltas["si_sdr"] >= request.minimum_si_sdr_gain_db
            else "rollback"
        )
    elif comparison.deltas:
        decision = "accept" if min(comparison.deltas.values()) >= 0 else "rollback"
    else:
        decision = "unverified" if request.allow_unverified else "rollback"
    selected = candidate if decision in {"accept", "unverified"} else source
    candidate_path = output_dir / "candidate.wav"
    output_path = output_dir / "enhanced.wav"
    report_path = output_dir / "audit.json"
    _write_audio(candidate_path, candidate, request.target_sample_rate)
    _write_audio(output_path, selected, request.target_sample_rate)
    report = {
        "schema_version": "lse.enhancement_run.v1",
        "created_at": utc_now(),
        "decision": decision,
        "source_audio": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "clean_reference": (
            str(request.clean_reference.expanduser().resolve())
            if request.clean_reference is not None
            else None
        ),
        "prescription_sha256": _sha256_bytes(request.prescription.encode("utf-8")),
        "plan": [asdict(action) for action in plan.actions],
        "thresholds": {"minimum_si_sdr_gain_db": request.minimum_si_sdr_gain_db},
        "metrics": comparison.to_dict(),
        "frame_metrics": frame_metrics,
        "artifacts": {
            "candidate_audio": str(candidate_path),
            "output_audio": str(output_path),
        },
        "claims": {
            "waveform_executed": True,
            "objective_gain_verified": decision == "accept",
            "rolled_back": decision == "rollback",
        },
    }
    write_json_atomic(report_path, report)
    return EnhancementRunResult(decision, output_path, candidate_path, report_path, comparison)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--prescription", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-reference", type=Path)
    parser.add_argument("--minimum-si-sdr-gain-db", type=float, default=0.0)
    parser.add_argument("--allow-unverified", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_enhancement(
        EnhancementRequest(
            audio_path=args.audio,
            prescription=args.prescription.read_text(encoding="utf-8"),
            output_dir=args.output_dir,
            clean_reference=args.clean_reference,
            minimum_si_sdr_gain_db=args.minimum_si_sdr_gain_db,
            allow_unverified=args.allow_unverified,
        )
    )
    print(
        json.dumps(
            {
                "decision": result.decision,
                "output_audio": str(result.output_audio),
                "audit_report": str(result.audit_report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
