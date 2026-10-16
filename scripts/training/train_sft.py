#!/usr/bin/env python3
"""Prepare legal/general instruction data or fine-tune a Qwen CPT checkpoint."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sft_trainer import main

if __name__ == '__main__':
    main()
