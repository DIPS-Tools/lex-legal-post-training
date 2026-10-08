#!/usr/bin/env python3
"""Evaluate the untouched Qwen3-8B base checkpoint on legal benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.runner import cli  # noqa: E402


DEFAULT_MODEL = Path("/home/mh1f25/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base")
DEFAULT_OUTPUT = PROJECT_ROOT / "results/evaluation/Qwen3-8B-Base"


if __name__ == "__main__":
    cli(
        default_model=DEFAULT_MODEL,
        default_output=DEFAULT_OUTPUT,
        default_label="Qwen3-8B-Base",
        description=__doc__ or "Evaluate the base model",
    )

