"""Render the README GIF from the committed 4090 stage matrix."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - documentation utility
    raise SystemExit("Install Pillow first: python -m pip install pillow") from exc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "readme" / "audio_policy_v2_demo.gif"
SIZE = (960, 540)
BG, PANEL, WHITE, MUTED = "#071015", "#101D25", "#F4FFF9", "#91A9A1"
GREEN, CYAN, AMBER, RED = "#5CF2A5", "#5CD6F2", "#FFD166", "#FF6B7A"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    color: str = WHITE,
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color)


def base(step: int, heading: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    for y in range(0, 540, 54):
        draw.line((0, y, 960, y), fill="#102128", width=1)
    label(draw, (42, 28), "AUDIO POLICY LAB", 24, WHITE, True)
    label(draw, (42, 61), "VERIFIABLE POST-TRAINING", 13, GREEN, True)
    label(draw, (800, 36), "v2.0.0", 16, CYAN, True)
    label(draw, (42, 108), f"0{step}  {heading}", 18, MUTED, True)
    draw.rounded_rectangle((42, 500, 918, 505), radius=3, fill="#173039")
    draw.rounded_rectangle((42, 500, 42 + int(876 * step / 6), 505), radius=3, fill=GREEN)
    return image, draw


def metric_card(draw: ImageDraw.ImageDraw, x: int, title: str, value: str, accent: str) -> None:
    draw.rounded_rectangle(
        (x, 220, x + 270, 335), radius=18, fill=PANEL, outline="#24404A", width=2
    )
    label(draw, (x + 20, 239), title, 14, MUTED, True)
    label(draw, (x + 20, 273), value, 24, accent, True)


def render() -> list[Image.Image]:
    matrix = json.loads((ROOT / "docs" / "stage_matrix_4090.json").read_text(encoding="utf-8"))
    stages = matrix["stages"]
    scenes = [
        (
            "ACOUSTIC EVIDENCE",
            ("SNR", "0 dB", CYAN),
            ("NOISE", "non-stationary", AMBER),
            ("RISK", "overprocess", RED),
        ),
        (
            "SFT",
            ("RECORDS", "114k train", CYAN),
            ("JSON", "1.000", GREEN),
            ("SCORE", f"{stages['sft']['metrics']['total']:.3f}", GREEN),
        ),
        (
            "CONSERVATIVE DPO",
            ("PAIR ACC", "1.000", GREEN),
            ("MARGIN", "2.1889", CYAN),
            ("REFERENCE", "explicit + frozen", AMBER),
        ),
        (
            "GRPO / RLVR",
            ("STEPS", "300 / 300", GREEN),
            ("MEAN REWARD", "0.9460", CYAN),
            ("ZERO-2", "engine verified", AMBER),
        ),
        (
            "PRESCRIPTION",
            ("DENOISE", "spectral gate", GREEN),
            ("STRENGTH", "bounded 0.42", CYAN),
            ("CONFIDENCE", "0.88", AMBER),
        ),
        (
            "HONEST EVAL",
            ("BASE", f"{stages['base']['metrics']['total']:.3f}", RED),
            ("SFT / DPO / GRPO", "0.964 tie", GREEN),
            ("CLAIM", "no fake uplift", AMBER),
        ),
    ]
    frames: list[Image.Image] = []
    for index, (heading, *cards) in enumerate(scenes, start=1):
        image, draw = base(index, heading)
        label(
            draw,
            (42, 151),
            "Evidence in.  Conservative DSP policy out.  Every field is testable.",
            21,
            WHITE,
            True,
        )
        for i, (title, value, accent) in enumerate(cards):
            metric_card(draw, 42 + i * 292, title, value, accent)
        draw.rounded_rectangle(
            (42, 374, 918, 458), radius=18, fill="#0D2726", outline="#246C58", width=2
        )
        label(
            draw,
            (66, 394),
            "Qwen2.5-1.5B + LoRA  /  single RTX 4090  /  same-ID evaluation",
            18,
            GREEN,
            True,
        )
        label(
            draw,
            (66, 426),
            "Format  |  diagnosis  |  bounds  |  consistency  |  overprocessing",
            14,
            MUTED,
        )
        frames.append(image)
    return frames


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = render()
    frames[0].save(
        OUTPUT, save_all=True, append_images=frames[1:], duration=1150, loop=0, optimize=True
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
