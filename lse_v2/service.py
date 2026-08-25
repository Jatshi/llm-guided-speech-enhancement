"""Authenticated FastAPI service for planning, batch enhancement, and PCM streaming."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .end_to_end import EnhancementRequest, run_enhancement
from .streaming import StreamingEnhancer


class Planner(Protocol):
    def plan(self, evidence: dict[str, Any]) -> str: ...


def _invoke_planner(planner: Any, audio_path: Path, evidence: dict[str, Any]) -> str:
    if hasattr(planner, "plan_audio"):
        prompt = (
            "Listen to the attached waveform and return one safe executable JSON prescription. "
            f"Observable evidence: {json.dumps(evidence, ensure_ascii=False)}"
        )
        return planner.plan_audio(audio_path, prompt)
    return planner.plan(evidence)


class RuleBasedPlanner:
    """Deterministic operational fallback; never presented as an LLM result."""

    def plan(self, evidence: dict[str, Any]) -> str:
        snr = evidence.get("snr_db")
        reduction = 12.0 if isinstance(snr, int | float) and snr < 10 else 7.0
        noise_type = str(evidence.get("noise_type") or "unknown")
        return json.dumps(
            {
                "diagnosis": {"noise_type": noise_type},
                "actions": [
                    {
                        "type": "spectral_subtraction",
                        "reduction_db": reduction,
                        "low_hz": 80,
                        "high_hz": 7600,
                    },
                    {"type": "limiter", "peak": 0.95},
                ],
                "rationale": "Conservative deterministic fallback from supplied evidence.",
                "confidence": 0.55,
            },
            ensure_ascii=False,
        )


class OpenAICompatiblePlanner:
    """Text-fallback backend for a vLLM/SGLang OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key

    def plan(self, evidence: dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one safe speech-enhancement prescription JSON object "
                            "with diagnosis, actions, rationale, confidence."
                        ),
                    },
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
                ],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            result = json.loads(response.read())
        text = result["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("planner response must be a JSON object")
        return json.dumps(parsed, ensure_ascii=False)


def authorize_token(authorization: str | None, expected_token: str | None) -> None:
    if expected_token is None:
        return
    if authorization != f"Bearer {expected_token}":
        raise PermissionError("invalid bearer token")


def _extract_basic_evidence(path: Path) -> dict[str, Any]:
    import soundfile as sf  # type: ignore[import-untyped]

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    frame = max(1, round(0.025 * sample_rate))
    energies = np.asarray(
        [np.mean(np.square(audio[index : index + frame])) for index in range(0, audio.size, frame)]
    )
    nonzero = energies[energies > 1e-10]
    snr = None
    if nonzero.size >= 4:
        snr = float(10 * np.log10(np.percentile(nonzero, 90) / np.percentile(nonzero, 10)))
    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": float(audio.size / sample_rate),
        "snr_db": round(snr, 3) if snr is not None else None,
        "noise_type": "unknown",
        "source": "direct_waveform_measurement",
    }


async def _save_upload(upload: Any, path: Path, max_bytes: int) -> None:
    received = 0
    with path.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            received += len(chunk)
            if received > max_bytes:
                raise ValueError(f"upload exceeds {max_bytes} bytes")
            handle.write(chunk)
    if received == 0:
        raise ValueError("uploaded audio is empty")


def create_app(planner: Planner | None = None, *, api_token: str | None = None):
    try:
        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket
        from fastapi.responses import FileResponse
        from starlette.background import BackgroundTask
    except ImportError as exc:
        raise RuntimeError("Install serving dependencies: pip install -e '.[serve]'") from exc

    selected = planner or RuleBasedPlanner()
    expected = api_token if api_token is not None else os.environ.get("LSE_API_TOKEN")
    max_upload_bytes = int(os.environ.get("LSE_MAX_UPLOAD_MIB", "50")) * 1024 * 1024
    app = FastAPI(title="LLM-Guided Speech Enhancement", version="4.0.0")

    def check(authorization: str | None) -> None:
        try:
            authorize_token(authorization, expected)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/prescriptions")
    async def prescriptions(
        audio: UploadFile = File(...),  # noqa: B008
        authorization: str | None = Header(default=None),  # noqa: B008
    ) -> dict[str, Any]:
        check(authorization)
        suffix = Path(audio.filename or "audio.wav").suffix
        with tempfile.TemporaryDirectory(prefix="lse-api-") as directory:
            path = Path(directory) / f"input{suffix}"
            try:
                await _save_upload(audio, path, max_upload_bytes)
            except ValueError as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            evidence = _extract_basic_evidence(path)
            prescription = _invoke_planner(selected, path, evidence)
        return {"evidence": evidence, "prescription": json.loads(prescription)}

    @app.post("/v1/enhance")
    async def enhance(
        audio: UploadFile = File(...),  # noqa: B008
        prescription: str | None = Form(default=None),  # noqa: B008
        authorization: str | None = Header(default=None),  # noqa: B008
    ):
        check(authorization)
        directory = Path(tempfile.mkdtemp(prefix="lse-enhance-"))
        path = directory / Path(audio.filename or "input.wav").name
        try:
            await _save_upload(audio, path, max_upload_bytes)
        except ValueError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        try:
            plan = prescription or _invoke_planner(selected, path, _extract_basic_evidence(path))
            result = run_enhancement(
                EnhancementRequest(
                    audio_path=path,
                    prescription=plan,
                    output_dir=directory / "output",
                    allow_unverified=True,
                )
            )
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return FileResponse(
            result.output_audio,
            media_type="audio/wav",
            headers={"X-LSE-Decision": result.decision, "X-LSE-Audit": str(result.audit_report)},
            background=BackgroundTask(shutil.rmtree, directory, ignore_errors=True),
        )

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        supplied = websocket.headers.get("authorization")
        try:
            authorize_token(supplied, expected)
        except PermissionError:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        config = json.loads(await websocket.receive_text())
        enhancer = StreamingEnhancer(
            json.dumps(config["prescription"]),
            sample_rate=int(config.get("sample_rate", 16000)),
            window_samples=int(config.get("window_samples", 8000)),
            hop_samples=int(config.get("hop_samples", 4000)),
        )
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") == "flush":
                await websocket.send_bytes(enhancer.flush().astype("<f4").tobytes())
                await websocket.close(code=1000)
                break
            data = message.get("bytes")
            if data:
                for chunk in enhancer.push(np.frombuffer(data, dtype="<f4")):
                    await websocket.send_bytes(chunk.astype("<f4").tobytes())

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--planner-url")
    parser.add_argument("--planner-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--native-stage-dir", type=Path)
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    parser.add_argument("--prefix-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    token = os.environ.get("LSE_API_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise RuntimeError("LSE_API_TOKEN is required when binding beyond loopback")
    if args.native_stage_dir and args.planner_url:
        raise ValueError("choose native-stage-dir or planner-url, not both")
    if args.native_stage_dir:
        from .native_inference import NativeAudioPlanner

        planner: Any = NativeAudioPlanner(
            args.native_stage_dir,
            whisper_model=args.whisper_model,
            language_model=args.planner_model,
            prefix_tokens=args.prefix_tokens,
        )
    elif args.planner_url:
        planner = OpenAICompatiblePlanner(
            args.planner_url,
            args.planner_model,
            os.environ.get("LSE_PLANNER_API_KEY"),
        )
    else:
        planner = RuleBasedPlanner()
    import uvicorn

    uvicorn.run(create_app(planner, api_token=token), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
