"""Backward-compatible offline reward evaluation entry point."""

from __future__ import annotations

from lse_v2.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
