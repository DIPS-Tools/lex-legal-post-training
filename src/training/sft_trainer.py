"""Full-parameter SFT with assistant-only loss and reusable prepared data."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path

from src.training.sft_data import (AssistantCollator, configure_tokenizer, file_hash,
                                   prepare, tokenizer_signature, write_json)

PROJECT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path('/home/mh1f25/scratch/llms_model/lexlegal_llms')
DATA_ROOT = Path('/home/mh1f25/scratch/legal_data_process/datasets')
DEFAULT_MODEL = MODEL_ROOT / 'cpt-model/Qwen-Lex-Legal-CPT-using-Legislation-3B'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--tokenizer-path', type=Path, help='Preparation tokenizer; defaults to --model-path')
    p.add_argument('--data-root', type=Path, default=DATA_ROOT / 'SFT-datasets')
    p.add_argument('--mmlu-root', type=Path, default=DATA_ROOT / 'evaluation/mmlu')
    p.add_argument('--prepared-dir', type=Path, default=DATA_ROOT / 'SFT-processed/LawInstruct-General-Balanced-Qwen3')
    p.add_argument('--prepare-only', action='store_true', help='CPU preparation; no model weights loaded')
    p.add_argument('--legal-examples', type=int, default=None,
                   help='Omit to use all eligible LawInstruct training examples after validation holdout')
    p.add_argument('--general-examples', type=int, default=None,
                   help='Omit to match the legal count; an explicit value must match it (50:50)')
    p.add_argument('--validation-fraction', type=float, default=0.01)
    p.add_argument('--validation-examples-per-domain', type=int, default=2000)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--max-records-per-source', type=int, default=0, help='Preparation smoke test only; 0 scans all')
    p.add_argument('--allow-smoke-data', action='store_true', help='Allow training on a limited preparation smoke subset')
    p.add_argument('--output-dir', type=Path, help='Default: sft-model/<input model name>-SFT')
    p.add_argument('--checkpoint-dir', type=Path, help='Default: checkpoints/<output folder name>')
    p.add_argument('--deepspeed', type=Path, default=PROJECT / 'configs/training/deepspeed_zero3.json')
    p.add_argument('--attn-implementation', choices=('flash_attention_2', 'sdpa', 'eager'), default='flash_attention_2')
    p.add_argument('--per-device-train-batch-size', type=int, default=1)
    p.add_argument('--per-device-eval-batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation-steps', type=int, default=16)
    p.add_argument('--num-train-epochs', type=float, default=1.0)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--warmup-ratio', type=float, default=0.03)
    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--eval-steps', type=int, default=250)
    p.add_argument('--save-steps', type=int, default=250)
    p.add_argument('--save-total-limit', type=int, default=2)
    p.add_argument('--dataloader-num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--report-to', nargs='+', default=['tensorboard'])
    p.add_argument('--resume-from-checkpoint', help='Path or auto (newest checkpoint with trainer_state.json)')
    args = p.parse_args(argv)
    for key in ('max_length', 'validation_examples_per_domain', 'per_device_train_batch_size',
                'per_device_eval_batch_size', 'gradient_accumulation_steps', 'num_train_epochs',
                'learning_rate', 'logging_steps', 'eval_steps', 'save_steps', 'save_total_limit'):
        if getattr(args, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if any(value is not None and value <= 0 for value in (args.legal_examples, args.general_examples)):
        p.error('Explicit example quotas must be positive')
    if args.legal_examples is not None and args.general_examples is not None and args.legal_examples != args.general_examples:
        p.error('Legal and general quotas must match for a 50:50 mixture')
    if not 0 < args.validation_fraction < 1 or not 0 <= args.warmup_ratio < 1:
        p.error('validation-fraction must be in (0,1); warmup-ratio in [0,1)')
    if args.max_records_per_source < 0 or args.dataloader_num_workers < 0 or args.max_steps == 0 or args.max_steps < -1:
        p.error('Invalid scan/worker/step limit')
    return args


def resume_path(value, checkpoint_dir):
    if value is None:
        return None
    if value == 'auto':
        choices = [p for p in checkpoint_dir.glob('checkpoint-*')
                   if p.name.split('-')[-1].isdigit() and (p / 'trainer_state.json').is_file()]
        if not choices:
            raise ValueError(f'No completed checkpoint state in {checkpoint_dir}')
        path = max(choices, key=lambda p: int(p.name.split('-')[-1]))
    else:
        path = Path(value).expanduser().resolve()
    if not (path / 'trainer_state.json').is_file():
        raise ValueError(f'Missing trainer_state.json in {path}; checkpoint may be incomplete')
    if not list(path.glob('global_step*/*optim_states.pt')):
        raise ValueError(f'Missing DeepSpeed optimizer state in {path}')
    return str(path)


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer
    if args.prepare_only:
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('Run --prepare-only with one CPU process, not torchrun')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, local_files_only=True, use_fast=True)
        prepare(args, tokenizer)
        return

    import torch
    import transformers
    from datasets import load_from_disk
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    if Version(transformers.__version__) < Version('4.51.0'):
        raise ValueError('Qwen3 requires transformers>=4.51.0')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('SFT requires a CUDA GPU with BF16 support')
    if 'LOCAL_RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    model_path = args.model_path.expanduser().resolve()
    out = (args.output_dir or MODEL_ROOT / 'sft-model' / (model_path.name + '-SFT')).expanduser().resolve()
    checkpoints = (args.checkpoint_dir or MODEL_ROOT / 'checkpoints' / out.name).expanduser().resolve()
    prepared = args.prepared_dir.expanduser().resolve()
    if out in (model_path, checkpoints, prepared) or checkpoints in (model_path, prepared):
        raise ValueError('Input model, prepared data, final output and checkpoints must be separate')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Final output is nonempty: {out}; choose a new output folder')
    manifest_path = prepared / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Run --prepare-only first: missing {manifest_path}')
    if not (model_path / 'config.json').is_file() or not args.deepspeed.is_file():
        raise ValueError('Missing local model config or DeepSpeed config')
    manifest = json.loads(manifest_path.read_text())
    if manifest['smoke_subset'] and not args.allow_smoke_data:
        raise ValueError('Prepared data is a smoke subset; pass --allow-smoke-data only for testing')
    if args.max_length != manifest['max_length']:
        raise ValueError('--max-length must match preparation manifest')
    tokenizer = AutoTokenizer.from_pretrained(prepared / 'tokenizer', local_files_only=True, use_fast=True)
    configure_tokenizer(tokenizer)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    configure_tokenizer(model_tokenizer)
    if tokenizer_signature(tokenizer) != manifest['tokenizer_signature'] or tokenizer_signature(model_tokenizer) != manifest['tokenizer_signature']:
        raise ValueError('Tokenizer differs from prepared data; use identical Qwen tokenization across checkpoints')
    del model_tokenizer
    data = load_from_disk(str(prepared / 'dataset'))
    if not len(data['train']) or not len(data['validation']):
        raise ValueError('Training and validation must both contain examples')
    checkpoints.mkdir(parents=True, exist_ok=True)
    resume = resume_path(args.resume_from_checkpoint, checkpoints)
    if resume is None and any(checkpoints.iterdir()):
        raise ValueError('Existing checkpoint output; resume explicitly or use a new directory')
    set_seed(args.seed)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world
    estimated_steps = args.max_steps if args.max_steps > 0 else math.ceil(math.ceil(len(data['train']) / effective_batch) * args.num_train_epochs)
    names = inspect.signature(TrainingArguments).parameters
    kwargs = dict(output_dir=str(checkpoints), do_train=True, do_eval=True,
                  eval_strategy='steps', eval_steps=args.eval_steps, save_strategy='steps',
                  save_steps=args.save_steps, save_total_limit=args.save_total_limit,
                  logging_steps=args.logging_steps, logging_first_step=True,
                  per_device_train_batch_size=args.per_device_train_batch_size,
                  per_device_eval_batch_size=args.per_device_eval_batch_size,
                  gradient_accumulation_steps=args.gradient_accumulation_steps,
                  num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
                  learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                  lr_scheduler_type='cosine', max_grad_norm=1.0,
                  warmup_steps=math.ceil(estimated_steps * args.warmup_ratio),
                  bf16=True, tf32=True, gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={'use_reentrant': False},
                  deepspeed=str(args.deepspeed.resolve()), prediction_loss_only=True,
                  dataloader_num_workers=args.dataloader_num_workers, seed=args.seed, data_seed=args.seed,
                  report_to=args.report_to, remove_unused_columns=False, ddp_find_unused_parameters=False)
    if 'save_safetensors' in names:
        kwargs['save_safetensors'] = True
    unsupported = set(kwargs) - set(names)
    if unsupported:
        raise ValueError(f'Unsupported TrainingArguments fields: {sorted(unsupported)}')
    # Must precede model loading to activate ZeRO-3 initialization.
    training_args = TrainingArguments(**kwargs)
    dtype_key = 'dtype' if Version(transformers.__version__).major >= 5 else 'torch_dtype'
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                attn_implementation=args.attn_implementation, **{dtype_key: torch.bfloat16})
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        raise ValueError('Prepared tokenizer exceeds model embedding vocabulary')
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    trainer_kwargs = dict(model=model, args=training_args, train_dataset=data['train'],
                          eval_dataset=data['validation'], data_collator=AssistantCollator(tokenizer.pad_token_id))
    if 'processing_class' in inspect.signature(Trainer).parameters:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    config = {'manifest_sha256': file_hash(manifest_path), 'model_path': str(model_path),
              'prepared_dir': str(prepared), 'final_output': str(out), 'world_size': world,
              'effective_batch_examples': effective_batch, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'training_arguments': training_args.to_dict(),
              'software': {'torch': torch.__version__, 'transformers': transformers.__version__}}
    config_path = checkpoints / 'sft_run_configuration.json'
    if resume:
        if not config_path.is_file():
            raise ValueError('Resume requires original sft_run_configuration.json in --checkpoint-dir')
        old = json.loads(config_path.read_text())
        for key in ('manifest_sha256', 'model_path', 'world_size', 'effective_batch_examples'):
            if old[key] != config[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        for key in ('learning_rate', 'num_train_epochs', 'max_steps', 'seed', 'max_length', 'warmup_ratio', 'weight_decay'):
            if old['arguments'][key] != config['arguments'][key]:
                raise ValueError(f'Resume training setting changed: {key}')
    elif trainer.is_world_process_zero():
        write_json(config_path, config)
    print(f'SFT: {len(data["train"]):,} train, {len(data["validation"]):,} validation; global batch {effective_batch}; resume={resume}', flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    metrics = trainer.evaluate()
    model.config.use_cache = True
    trainer.save_model(str(out))  # All ranks participate in ZeRO-3 gathering.
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(out)
        write_json(out / 'sft_run_configuration.json', config)
        write_json(out / 'sft_training_summary.json', {'global_step': trainer.state.global_step,
                   'train_metrics': result.metrics, 'validation_metrics': metrics,
                   'data_statistics': manifest['statistics']})
        print(f'Saved SFT model: {out}', flush=True)
