# Recording the Demo GIF

The README hero image (`assets/readme/hero-ui.png`) is a **UI concept mock**.
To replace the animated placeholder (`assets/readme/demo.gif`) with a real
screen recording of the live Gradio app, follow the steps below.

## 1. Launch the demo

```bash
bash scripts/start_demo.sh
# Gradio serves on http://0.0.0.0:6006  (share link also printed)
```

## 2. Record the screen

Pick any recorder and export to GIF:

- **Windows** — [ScreenToGif](https://www.screentogif.com/) (free, direct GIF export).
- **macOS** — [Kap](https://getkap.co/) or QuickTime + `ffmpeg` conversion.
- **Linux** — [Peek](https://github.com/phw/peek) or `ffmpeg -f x11grab`.

Convert a captured `mp4` to an optimized GIF:

```bash
ffmpeg -i demo.mp4 -vf "fps=12,scale=960:-1:flags=lanczos" -loop 0 assets/readme/demo.gif
# Optional: shrink with gifsicle
gifsicle -O3 --colors 128 assets/readme/demo.gif -o assets/readme/demo.gif
```

## 3. Recommended shot list (≤ 20s)

1. Upload / drop a noisy clip → waveform + spectrogram render.
2. Click **Analyze** → the three-section LLM output (Diagnosis / Strategy / Rationale) streams in.
3. Click **Enhance** → before/after waveform comparison + play the cleaned audio.

## 4. Wire it into the README

The README already references `assets/readme/demo.gif`. Once the file exists,
it renders automatically at the top of the page — no markup changes needed.

> Keep the GIF under ~10 MB so GitHub renders it inline without lazy-loading.
