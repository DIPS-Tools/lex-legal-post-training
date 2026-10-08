"""Full-parameter causal-language-model continued pretraining for Qwen3."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cpt_data import CptDataError, PackedTokenDataset, load_metadata, write_json  # noqa: E402


DEFAULT_MODEL = Path("/home/mh1f25/scratch/llms_model/lexlegal_llms/base/Qwen3-8B-Base")
DEFAULT_TOKENIZED = Path(
    "/home/mh1f25/scratch/legal_data_process/datasets/CPT-tokenized/CPT-Proportional-5B-Qwen3"
)
DEFAULT_FINAL_OUTPUT = Path(
    "/home/mh1f25/scratch/llms_model/lexlegal_llms/cpt-model/"
    "Qwen-Lex-Legal-CPT-using-Proportional-5B"
)
DEFAULT_CHECKPOINTS = Path(
    "/home/mh1f25/scratch/llms_model/lexlegal_llms/checkpoints/"
    "Qwen-Lex-Legal-CPT-using-Proportional-5B"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenized-dir", type=Path, default=DEFAULT_TOKENIZED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FINAL_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--deepspeed", type=Path, default=PROJECT_ROOT / "configs/training/deepspeed_zero3.json")
    parser.add_argument("--sequence-length", type=int, default=4096, choices=(4096, 8192))
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--target-training-tokens",
        type=int,
        default=None,
        help="Token budget; defaults to one pass over all complete training blocks.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, or 'auto' to use the latest checkpoint in --checkpoint-dir.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--report-to", nargs="+", default=["tensorboard"])
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--overwrite-final-output", action="store_true")
    return parser.parse_args()


def require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise CptDataError(f"{name} must be positive, got {value}")


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < math.log(sys.float_info.max) else math.inf


def optional_dataset(path: Path, sequence_length: int) -> PackedTokenDataset | None:
    if not path.is_file() or path.stat().st_size < sequence_length * 4:
        return None
    return PackedTokenDataset(path, sequence_length)


def resolve_resume_checkpoint(value: str | None, checkpoint_dir: Path) -> str | None:
    if value is None:
        return None
    if value != "auto":
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise CptDataError(f"Resume checkpoint does not exist: {path}")
        return str(path)
    from transformers.trainer_utils import get_last_checkpoint

    checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.is_dir() else None
    if checkpoint is None:
        raise CptDataError(f"No checkpoint found for automatic resumption in {checkpoint_dir}")
    return checkpoint


def main() -> None:
    args = parse_args()
    for name in (
        "sequence_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
    ):
        require_positive(name, getattr(args, name))

    try:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    except ImportError as exc:
        raise RuntimeError("torch, transformers>=4.51.0, and packaging are required for CPT") from exc
    if Version(transformers.__version__) < Version("4.51.0"):
        raise RuntimeError(f"Qwen3 requires transformers>=4.51.0; found {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CPT requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The allocated GPU does not support BF16")

    model_path = args.model_path.expanduser().resolve()
    tokenized_dir = args.tokenized_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    deepspeed_path = args.deepspeed.expanduser().resolve()
    for required in (model_path / "config.json", tokenized_dir / "metadata.json", deepspeed_path):
        if not required.is_file():
            raise CptDataError(f"Required file is missing: {required}")
    if output_dir == checkpoint_dir:
        raise CptDataError("--output-dir and --checkpoint-dir must be different directories")
    if (output_dir / "config.json").exists() and not args.overwrite_final_output:
        raise CptDataError(
            f"Final model already exists at {output_dir}; choose another --output-dir or pass --overwrite-final-output"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from_checkpoint is None and any(checkpoint_dir.glob("checkpoint-*")):
        raise CptDataError(
            f"Existing checkpoints found in {checkpoint_dir}; pass --resume-from-checkpoint auto or use a new directory"
        )

    metadata = load_metadata(tokenized_dir)
    train_dataset = PackedTokenDataset(tokenized_dir / "train_tokens.bin", args.sequence_length)
    eval_datasets: dict[str, PackedTokenDataset] = {}
    if not args.no_eval:
        for name, filename in (
            ("all", "validation_tokens.bin"),
            ("legal", "validation_legal_tokens.bin"),
            ("replay", "validation_replay_tokens.bin"),
        ):
            dataset = optional_dataset(tokenized_dir / filename, args.sequence_length)
            if dataset is not None:
                eval_datasets[name] = dataset
        if "all" not in eval_datasets:
            raise CptDataError("Validation data has no complete sequence; use --no-eval only for a smoke test")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_tokens_per_step = (
        args.sequence_length
        * args.per_device_train_batch_size
        * world_size
        * args.gradient_accumulation_steps
    )
    usable_train_tokens = len(train_dataset) * args.sequence_length
    target_training_tokens = args.target_training_tokens or usable_train_tokens
    require_positive("target_training_tokens", target_training_tokens)
    calculated_steps = math.ceil(target_training_tokens / global_tokens_per_step)
    max_steps = args.max_steps if args.max_steps is not None else calculated_steps
    require_positive("max_steps", max_steps)
    planned_tokens = max_steps * global_tokens_per_step

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared_tokenizer = metadata.get("tokenizer", {})
    for key, actual in (("eos_token_id", tokenizer.eos_token_id), ("vocab_size", len(tokenizer))):
        expected = prepared_tokenizer.get(key)
        if expected is not None and expected != actual:
            raise CptDataError(f"Tokenizer {key} mismatch: tokenized data={expected}, model tokenizer={actual}")

    print(
        json.dumps(
            {
                "world_size": world_size,
                "sequence_length": args.sequence_length,
                "global_tokens_per_step": global_tokens_per_step,
                "usable_train_tokens": usable_train_tokens,
                "target_training_tokens": target_training_tokens,
                "max_steps": max_steps,
                "planned_tokens_including_final_step_overshoot": planned_tokens,
                "eval_splits": {name: len(dataset) for name, dataset in eval_datasets.items()},
            },
            indent=2,
        ),
        flush=True,
    )

    run_name = args.run_name or output_dir.name
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=bool(eval_datasets),
        eval_strategy="steps" if eval_datasets else "no",
        eval_steps=args.eval_steps if eval_datasets else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_steps=max_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=str(deepspeed_path),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        report_to=args.report_to,
        run_name=run_name,
        seed=args.seed,
        data_seed=args.data_seed,
        remove_unused_columns=False,
        save_safetensors=True,
        ddp_find_unused_parameters=False,
    )
    # Construct TrainingArguments before loading the model so Transformers can
    # activate DeepSpeed ZeRO-3 initialization during from_pretrained().
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets or None,
        data_collator=default_data_collator,
    )

    run_configuration: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(args.project_root.expanduser().resolve()),
        "model_path": str(model_path),
        "model_config_commit_hash": getattr(model.config, "_commit_hash", None),
        "tokenized_dir": str(tokenized_dir),
        "tokenized_metadata": metadata,
        "checkpoint_dir": str(checkpoint_dir),
        "final_output_dir": str(output_dir),
        "training_arguments": training_args.to_dict(),
        "token_accounting": {
            "world_size": world_size,
            "global_tokens_per_step": global_tokens_per_step,
            "usable_train_tokens": usable_train_tokens,
            "target_training_tokens": target_training_tokens,
            "planned_tokens": planned_tokens,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    if trainer.is_world_process_zero():
        write_json(checkpoint_dir / "run_configuration.json", run_configuration)

    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, checkpoint_dir)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)

    evaluation_metrics: dict[str, Any] = {}
    if eval_datasets:
        evaluation_metrics = trainer.evaluate(eval_dataset=eval_datasets)
        for key, value in list(evaluation_metrics.items()):
            if key.endswith("_loss") and isinstance(value, (float, int)):
                evaluation_metrics[key.removesuffix("_loss") + "_perplexity"] = perplexity(float(value))

    final_summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_step": trainer.state.global_step,
        "actual_tokens_processed_including_step_overshoot": trainer.state.global_step * global_tokens_per_step,
        "train_metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }
    if trainer.is_world_process_zero():
        write_json(output_dir / "cpt_training_summary.json", final_summary)
        write_json(output_dir / "cpt_run_configuration.json", run_configuration)
    print(json.dumps(final_summary, indent=2, default=str), flush=True)


def cli() -> None:
    try:
        main()
    except (CptDataError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    cli()
