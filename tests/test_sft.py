"""SFT regression tests: source formats, leakage, masking, and reusable artifacts."""
import json
import lzma
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.training.sft_data import (CHAT_TEMPLATE, MmluGuard, SOURCES, SUBJECTS,
    allocate, configure_tokenizer, encode_messages, normalize_record, prepare,
    read_rows, tokenizer_signature, AssistantCollator, resolve_training_targets)
from src.training.sft_trainer import parse_args, resume_path


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.BPE(unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(['system user assistant hello yes no legal question answer 0123456789'],
        trainers.BpeTrainer(vocab_size=300, special_tokens=['[UNK]', '<|im_start|>', '<|im_end|>'],
                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
    configure_tokenizer(tok)
    return tok


class SftTests(unittest.TestCase):
    def test_automatic_balanced_targets(self):
        args = parse_args([])
        self.assertIsNone(args.legal_examples)
        self.assertIsNone(args.general_examples)
        self.assertEqual(resolve_training_targets(None, None, {'legal': 37, 'general': 100}),
                         {'legal': 37, 'general': 37})
        self.assertEqual(resolve_training_targets(20, None, {'legal': 37, 'general': 100}),
                         {'legal': 20, 'general': 20})
        with self.assertRaisesRegex(ValueError, 'shortfall 7'):
            resolve_training_targets(None, None, {'legal': 37, 'general': 30})
        with self.assertRaisesRegex(ValueError, 'equal quotas'):
            resolve_training_targets(None, 10, {'legal': 37, 'general': 100})
        with self.assertRaisesRegex(ValueError, 'both pools'):
            resolve_training_targets(None, None, {'legal': 0, 'general': 100})

    def test_adapters(self):
        law = dict(dataset_name='Example', subset_name='qa', instruction_language='en',
            prompt_language='en', answer_language='en', instruction='Answer', prompt='Question?', answer='Yes')
        record = normalize_record('lawinstruct', law)
        self.assertEqual(record['messages'][0]['content'], 'Answer\n\nQuestion?')
        with self.assertRaisesRegex(ValueError, 'non_english'):
            normalize_record('lawinstruct', dict(law, instruction_language='nl'))
        for source, row in [
            ('SlimOrca', {'conversations': [{'from':'human','value':'Q'}, {'from':'gpt','value':'A'}]}),
            ('MetaMathQA', {'query':'Q', 'response':'A'}),
            ('glaive-code-assistant-v2', {'question':'Q', 'answer':'A'}),
            ('ultrachat_200k', {'messages':[{'role':'user','content':'Q'}, {'role':'assistant','content':'A'}]})]:
            self.assertEqual(normalize_record(source,row)['messages'][-1]['content'], 'A')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            normalize_record('ultrachat_200k', {'messages':[{'role':'user','content':'Q'}]})

    def test_mask_matches_template_and_all_assistant_turns(self):
        tok = tiny_tokenizer()
        messages = [{'role':'system','content':'System policy'}, {'role':'user','content':'First question'},
                    {'role':'assistant','content':'Yes'}, {'role':'user','content':'Second question'},
                    {'role':'assistant','content':'No'}]
        row = encode_messages(messages, tok, 4096)
        expected = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
        self.assertEqual(row['input_ids'], expected)
        supervised = tok.decode([x for x in row['labels'] if x != -100])
        self.assertEqual(supervised, 'Yes<|im_end|>No<|im_end|>')
        self.assertEqual(row['labels'][0], -100)
        with self.assertRaisesRegex(ValueError, 'overlength'):
            encode_messages(messages, tok, 5)
        with tempfile.TemporaryDirectory() as directory:
            from transformers import AutoTokenizer
            before = tokenizer_signature(tok)
            tok.save_pretrained(directory)
            restored = AutoTokenizer.from_pretrained(directory)
            configure_tokenizer(restored)
            self.assertEqual(before, tokenizer_signature(restored))
            self.assertEqual(restored.chat_template, CHAT_TEMPLATE)

    def test_padding_does_not_mask_real_eos(self):
        tok = tiny_tokenizer()
        rows = [encode_messages([{'role':'user','content':q}, {'role':'assistant','content':'Yes'}],tok,4096)
                for q in ('Q', 'A longer question')]
        batch = AssistantCollator(tok.pad_token_id)(rows)
        for index, row in enumerate(rows):
            self.assertEqual(batch['labels'][index,:len(row['labels'])].tolist(), row['labels'])
            self.assertTrue(all(x == -100 for x in batch['labels'][index,len(row['labels']):].tolist()))
            self.assertIn(tok.eos_token_id, row['labels'])

    def test_allocation_and_resume_guards(self):
        self.assertEqual(allocate({'a':3,'b':2},4), {'a':2,'b':2})
        with self.assertRaises(ValueError):
            allocate({'a':3},4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10,20):
                (root / f'checkpoint-{step}').mkdir()
            checkpoint = root/'checkpoint-10'
            (checkpoint/'trainer_state.json').write_text('{}')
            (checkpoint/'global_step10').mkdir()
            (checkpoint/'global_step10/zero_pp_rank_0_optim_states.pt').write_bytes(b'test fixture')
            self.assertEqual(resume_path('auto', root), str(checkpoint))

    def test_tiny_qwen_training_and_reload(self):
        """Exercise real causal loss/backward, Trainer, collation and model export on CPU."""
        import inspect
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, AutoTokenizer
        tok = tiny_tokenizer()
        rows = [encode_messages([{'role':'user','content':'Question'},
                {'role':'assistant','content':answer}], tok, 4096) for answer in ('Yes','No')]
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id))
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=2,
                learning_rate=1e-3, report_to=[], save_strategy='no', disable_tqdm=True,
                dataloader_pin_memory=False)
            kwargs = {'processing_class' if 'processing_class' in inspect.signature(Trainer).parameters else 'tokenizer':tok}
            trainer = Trainer(model=model, args=args, train_dataset=rows,
                data_collator=AssistantCollator(tok.pad_token_id), **kwargs)
            result = trainer.train()
            self.assertTrue(math.isfinite(result.training_loss))
            self.assertEqual(trainer.state.global_step,2)
            trainer.save_model(directory)
            tok.save_pretrained(directory)
            reloaded = Qwen3ForCausalLM.from_pretrained(directory)
            loaded_tok = AutoTokenizer.from_pretrained(directory)
            self.assertEqual(loaded_tok.chat_template, CHAT_TEMPLATE)
            self.assertEqual(reloaded.config.eos_token_id, loaded_tok.eos_token_id)
            with torch.no_grad():
                loss = reloaded(**AssistantCollator(tok.pad_token_id)(rows)).loss
            self.assertTrue(torch.isfinite(loss).item())

    def test_end_to_end_preparation(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_from_disk
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, mmlu, out = root/'raw', root/'mmlu', root/'prepared'
            question = 'Under which conditions can a federal tribunal exercise exclusive jurisdiction over this proceeding'
            for subject in SUBJECTS:
                (mmlu/subject).mkdir(parents=True)
                for split in ('dev','validation','test'):
                    (mmlu/subject/f'{split}.jsonl').write_text(json.dumps({'question':question})+'\n')
            guard = MmluGuard(mmlu)
            self.assertIsNotNone(guard.match('Please answer: ' + question + '?'))
            self.assertIsNone(guard.match('Explain how to bake bread'))
            paths = {'lawinstruct':'data/example-train-0.jsonl.xz',
                     'SlimOrca':'oo-labeled_correct.gpt4.sharegpt.jsonl',
                     'MetaMathQA':'MetaMathQA-395K.json',
                     'glaive-code-assistant-v2':'glaive_code_assistant_v2.json',
                     'ultrachat_200k':'data/train_sft-00000.parquet'}
            for source in SOURCES:
                path = raw/source/paths[source]
                path.parent.mkdir(parents=True)
                rows = []
                for i in range(40):
                    prompt = f'{source} distinct task number {i}' if i else question
                    if source == 'lawinstruct':
                        row = dict(dataset_name='Example', subset_name='qa', instruction_language='en',
                            prompt_language='en', answer_language='en', instruction='Answer', prompt=prompt, answer='Yes')
                    elif source == 'SlimOrca':
                        row = {'conversations':[{'from':'human','value':prompt},{'from':'gpt','value':'Yes'}]}
                    elif source == 'MetaMathQA':
                        row = {'query':prompt,'response':'Yes'}
                    elif source == 'glaive-code-assistant-v2':
                        row = {'question':prompt,'answer':'Yes'}
                    else:
                        row = {'messages':[{'role':'user','content':prompt},{'role':'assistant','content':'Yes'}]}
                    rows.append(row)
                if path.suffix == '.xz':
                    with lzma.open(path,'wt') as handle:
                        handle.write('\n'.join(map(json.dumps,rows))+'\n')
                elif path.suffix == '.parquet':
                    pq.write_table(pa.Table.from_pylist(rows),path)
                    # This deliberately invalid excluded split must never be read.
                    (path.parent/'test_sft-00000.parquet').write_bytes(b'not training data')
                elif path.suffix == '.json':
                    path.write_text(json.dumps(rows))
                else:
                    path.write_text('\n'.join(map(json.dumps,rows))+'\n')
            args = parse_args(['--prepare-only','--data-root',str(raw),'--mmlu-root',str(mmlu),
                '--prepared-dir',str(out),'--legal-examples','10','--general-examples','10',
                '--validation-fraction','0.25','--validation-examples-per-domain','4'])
            manifest = prepare(args,tiny_tokenizer())
            data = load_from_disk(str(out/'dataset'))
            self.assertEqual(len(data['train']),20)
            self.assertEqual(len(data['validation']),8)
            self.assertEqual(manifest['statistics']['train']['legal/examples'],10)
            self.assertEqual(manifest['statistics']['train']['general/examples'],10)
            train = list(read_rows(out/'train.jsonl'))
            validation = list(read_rows(out/'validation.jsonl'))
            self.assertFalse({r['group'] for r in train} & {r['group'] for r in validation})
            self.assertFalse({r['prompt_hash'] for r in train} & {r['prompt_hash'] for r in validation})
            self.assertEqual(sum(v for k,v in manifest['counts'].items() if k.endswith('/mmlu_overlap')),5)
            with self.assertRaisesRegex(ValueError,'empty'):
                prepare(args,tiny_tokenizer())
            args.prepared_dir = root / 'prepared-all'
            args.legal_examples = None
            args.general_examples = None
            automatic = prepare(args, tiny_tokenizer())
            eligible = sum(automatic['quotas']['legal/train']['available'].values())
            self.assertGreater(eligible, 10)
            self.assertEqual(automatic['statistics']['train']['legal/examples'], eligible)
            self.assertEqual(automatic['statistics']['train']['general/examples'], eligible)
            self.assertEqual(automatic['quotas']['legal/train']['selected'],
                             automatic['quotas']['legal/train']['available'])


if __name__ == '__main__':
    unittest.main()
