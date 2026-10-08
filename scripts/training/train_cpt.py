#!/usr/bin/env python3
"""Command-line entry point for Qwen3 continued pretraining."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cpt_trainer import cli  # noqa: E402


if __name__ == "__main__":
    cli()
