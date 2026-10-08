#!/usr/bin/env python3
"""Evaluate the legal-CPT Qwen3-8B checkpoint on legal benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.runner import cli  # noqa: E402


DEFAULT_MODEL = Path(
    "/home/mh1f25/scratch/llms_model/lexlegal_llms/cpt-model/"
    "Qwen-Lex-Legal-CPT-using-Proportional-5B"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/evaluation/Qwen3-8B-CPT-Proportional-5B"


if __name__ == "__main__":
    cli(
        default_model=DEFAULT_MODEL,
        default_output=DEFAULT_OUTPUT,
        default_label="Qwen3-8B-CPT-Proportional-5B",
        description=__doc__ or "Evaluate the continued-pretrained model",
    )

