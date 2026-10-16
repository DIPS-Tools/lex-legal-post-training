"""Streaming SFT adapters, reproducible sampling, and assistant-only Qwen labels."""
from __future__ import annotations

import hashlib
import json
import lzma
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SOURCES = ('lawinstruct', 'SlimOrca', 'MetaMathQA', 'ultrachat_200k', 'glaive-code-assistant-v2')
SUBJECTS = ('international_law', 'professional_law', 'jurisprudence')
CHAT_TEMPLATE = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def clean(value):
    if not isinstance(value, str):
        raise ValueError('Expected string content')
    value = unicodedata.normalize('NFKC', value).replace('\r\n', '\n').replace('\r', '\n')
    return ''.join(c for c in value if c in '\n\t' or not unicodedata.category(c).startswith('C')).strip()


def words(value):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', value).casefold())


def read_rows(path):
    """Read local files only; JSON arrays stream through ijson."""
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif path.suffix == '.json':
        import ijson
        with path.open('rb') as handle:
            yield from ijson.items(handle, 'item')
    else:
        opener = lzma.open if path.suffix == '.xz' else open
        with opener(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def source_files(root):
    root = Path(root)
    patterns = {
        'lawinstruct': 'data/*-train-*.jsonl.xz',
        'SlimOrca': 'oo-labeled_correct.gpt4.sharegpt.jsonl',
        'MetaMathQA': 'MetaMathQA-395K.json',
        'ultrachat_200k': 'data/train_sft-*.parquet',
        'glaive-code-assistant-v2': 'glaive_code_assistant_v2.json',
    }
    result = {}
    for source, pattern in patterns.items():
        result[source] = sorted((root / source).glob(pattern))
        if not result[source]:
            raise ValueError(f'Missing training files: {root / source / pattern}')
    return result


def normalize_record(source, row):
    subset = source
    group_text = None
    if source == 'lawinstruct':
        if any(row.get(key) != 'en' for key in ('instruction_language', 'prompt_language', 'answer_language')):
            raise ValueError('non_english')
        subset = str(row['dataset_name']) + '/' + str(row['subset_name'])
        if any(name in subset.casefold() for name in ('mmlu', *SUBJECTS)):
            raise ValueError('mmlu_source')
        prompt = clean(row['prompt'])
        messages = [{'role': 'user', 'content': clean(row['instruction']) + '\n\n' + prompt},
                    {'role': 'assistant', 'content': row['answer']}]
        group_text = prompt  # Same task prompt with different instructions stays together.
    elif source == 'SlimOrca':
        roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
        messages = [{'role': roles[m['from']], 'content': m['value']} for m in row['conversations']]
    elif source == 'ultrachat_200k':
        messages = row['messages']
    else:
        question, answer = ('query', 'response') if source == 'MetaMathQA' else ('question', 'answer')
        messages = [{'role': 'user', 'content': row[question]}, {'role': 'assistant', 'content': row[answer]}]
        group_text = row.get('original_question')
    messages = [{'role': m['role'], 'content': clean(m['content'])} for m in messages]
    expected = 'user'
    for i, message in enumerate(messages):
        if not message['content']:
            raise ValueError('empty_message')
        if any(token in message['content'] for token in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
            raise ValueError('embedded_chat_delimiter')
        if i == 0 and message['role'] == 'system':
            continue
        if message['role'] != expected:
            raise ValueError('invalid_role_sequence')
        expected = 'assistant' if expected == 'user' else 'user'
    if not messages or expected != 'user' or messages[-1]['role'] != 'assistant':
        raise ValueError('incomplete_conversation')
    users = '\n\n'.join(m['content'] for m in messages if m['role'] == 'user')
    return {'source': source, 'subset': subset, 'messages': messages,
            'group': digest(' '.join(words(group_text or users))),
            'prompt_hash': digest(' '.join(words(users)))}


def configure_tokenizer(tokenizer):
    if not tokenizer.is_fast:
        raise ValueError('Assistant masking requires a fast tokenizer with character offsets')
    for marker in ('<|im_start|>', '<|im_end|>'):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != marker:
            raise ValueError(f'Model tokenizer lacks existing Qwen ChatML token {marker}')
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.eos_token = '<|im_end|>'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'


def encode_messages(messages, tokenizer, max_length):
    """Tokenize once; label only assistant content and its end-of-turn token.

    No packing or silent truncation: overlength conversations are rejected.
    Offset containment avoids accidentally supervising user/header tokens.
    """
    text, spans = '', []
    for message in messages:
        text += '<|im_start|>' + message['role'] + '\n'
        start = len(text)
        text += message['content'] + '<|im_end|>'
        if message['role'] == 'assistant':
            spans.append((start, len(text)))
        text += '\n'
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if len(ids) > max_length:
        raise ValueError('overlength')
    labels = [token if end > start and any(lo <= start and end <= hi for lo, hi in spans) else -100
              for token, (start, end) in zip(ids, encoded['offset_mapping'], strict=True)]
    if not any(value != -100 for value in labels[1:]):
        raise ValueError('no_assistant_tokens')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def tokenizer_signature(tokenizer):
    # Includes the tokenization algorithm as well as vocabulary and special IDs.
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # These mutable runtime settings change after calling a fast tokenizer.
    backend.pop('padding', None)
    backend.pop('truncation', None)
    return digest(json.dumps(backend, sort_keys=True) + CHAT_TEMPLATE)


class MmluGuard:
    """Remove >=80% five-word-shingle containment of any held-out question.

    This lexical check is not a semantic paraphrase detector. All local dev,
    validation and test shards of the three legal subjects are indexed.
    """
    def __init__(self, root):
        self.index = defaultdict(set)
        self.widths = set()
        self.sizes, self.files = {}, []
        for subject in SUBJECTS:
            for split in ('dev', 'validation', 'test'):
                paths = sorted((Path(root) / subject).glob(split + '*.parquet'))
                if not paths:
                    paths = sorted((Path(root) / subject).glob(split + '*.jsonl'))
                if not paths:
                    raise ValueError(f'Missing MMLU guard input: {subject}/{split}')
                for path in paths:
                    self.files.append({'path': str(path), 'sha256': file_hash(path)})
                    for i, row in enumerate(read_rows(path)):
                        tokens = words(row['question'])
                        n = min(5, len(tokens))
                        if not n:
                            continue
                        self.widths.add(n)
                        key = f'{subject}/{split}/{path.name}/{i}'
                        shingles = set(tuple(tokens[j:j+n]) for j in range(len(tokens)-n+1))
                        self.sizes[key] = len(shingles)
                        for shingle in shingles:
                            self.index[shingle].add(key)

    def match(self, text):
        tokens = words(text)
        hits = Counter()
        for n in sorted(self.widths):
            for shingle in set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)):
                hits.update(self.index.get(shingle, ()))
        return next((key for key, count in sorted(hits.items()) if count / self.sizes[key] >= 0.8), None)


def allocate(counts, target):
    """Proportional quotas with deterministic largest-remainder rounding."""
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f'Requested {target:,} examples, only {total:,} eligible')
    if target == 0:
        return {key: 0 for key in counts}
    quotas = {key: count * target // total for key, count in counts.items()}
    order = sorted(counts, key=lambda key: (-(counts[key] * target % total), key))
    for key in order[:target - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def resolve_training_targets(legal, general, available):
    """Resolve automatic quotas after filtering/holdout; never sample with replacement."""
    legal = available.get('legal', 0) if legal is None else legal
    general = legal if general is None else general
    if legal <= 0 or general <= 0:
        raise ValueError('A balanced mixture requires eligible examples in both pools')
    if legal != general:
        raise ValueError(f'50:50 requires equal quotas: legal={legal:,}, general={general:,}')
    for domain, target in (('legal', legal), ('general', general)):
        count = available.get(domain, 0)
        if target > count:
            raise ValueError(
                f'{domain} quota {target:,} exceeds {count:,} eligible training examples '
                f'(shortfall {target-count:,}). Cannot use all legal data and keep 50:50 '
                f'without repetition. Add more eligible data or set --legal-examples '
                f'{min(available.values()):,} (without commas) for a smaller balanced sample.'
            )
    return {'legal': legal, 'general': general}


def prepare(args, tokenizer):
    """Single-process CPU preparation; completion manifest is written last."""
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
    out = args.prepared_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f'Preparation directory must be empty: {out}; use a new directory')
    guard = MmluGuard(args.mmlu_root)
    files = source_files(args.data_root)
    configure_tokenizer(tokenizer)
    tokenizer.save_pretrained(out / 'tokenizer')
    database = out / 'candidates.sqlite'
    counts = Counter()
    conn = sqlite3.connect(database)
    conn.execute('PRAGMA temp_store=FILE')
    conn.execute('CREATE TABLE examples (id TEXT PRIMARY KEY, prompt TEXT, response TEXT, grp TEXT, domain TEXT, stratum TEXT, split TEXT, priority TEXT, payload TEXT, tokens TEXT)')
    inputs = []
    with (out / 'rejections.jsonl').open('w', encoding='utf-8') as rejected:
        for source, paths in files.items():
            seen = 0
            for path in paths:
                inputs.append({'path': str(path), 'sha256': file_hash(path)})
                print(f'Preparing {path.name}', flush=True)
                for row_number, row in enumerate(read_rows(path)):
                    if args.max_records_per_source and seen >= args.max_records_per_source:
                        break
                    seen += 1
                    counts[source + '/scanned'] += 1
                    try:
                        record = normalize_record(source, row)
                        evidence = guard.match('\n'.join(m['content'] for m in record['messages']))
                        if evidence:
                            raise ValueError('mmlu_overlap:' + evidence)
                        encoded = encode_messages(record['messages'], tokenizer, args.max_length)
                    except (ValueError, KeyError, TypeError) as exc:
                        reason = str(exc)
                        counts[source + '/rejected/' + reason.split(':')[0]] += 1
                        rejected.write(json.dumps({'source': source, 'file': path.name, 'row': row_number, 'reason': reason}) + '\n')
                        continue
                    record['file'], record['row'] = path.name, row_number
                    record['metadata'] = {key: row[key] for key in ('jurisdiction', 'task_type', 'source', 'type') if key in row}
                    response = digest(json.dumps([m['content'] for m in record['messages'] if m['role'] == 'assistant']))
                    identity = digest(record['prompt_hash'] + response)
                    split = 'validation' if int(digest(str(args.seed) + record['group'])[:16], 16) / 2**64 < args.validation_fraction else 'train'
                    priority = digest(str(args.seed) + identity)
                    domain = 'legal' if source == 'lawinstruct' else 'general'
                    before = conn.total_changes
                    conn.execute('INSERT OR IGNORE INTO examples VALUES (?,?,?,?,?,?,?,?,?,?)',
                                 (identity, record['prompt_hash'], response, record['group'], domain,
                                  record['subset'], split, priority, json.dumps(record, ensure_ascii=False), json.dumps(encoded)))
                    counts[source + ('/unique' if conn.total_changes > before else '/duplicate')] += 1
                    if seen % 10000 == 0:
                        conn.commit()
                        print(f'{source}: scanned={seen:,}', flush=True)
                conn.commit()
                if args.max_records_per_source and seen >= args.max_records_per_source:
                    break
    # Keep no arbitrary target when the same normalized prompt has different answers.
    conn.execute('CREATE INDEX prompt_idx ON examples(prompt)')
    conn.execute('CREATE TEMP TABLE conflicts AS SELECT prompt FROM examples GROUP BY prompt HAVING COUNT(DISTINCT response)>1')
    counts['conflicting_prompts'] = conn.execute('SELECT COUNT(*) FROM conflicts').fetchone()[0]
    with (out / 'conflicting_prompts.jsonl').open('w') as handle:
        for (identity,) in conn.execute('SELECT id FROM examples WHERE prompt IN (SELECT prompt FROM conflicts)'):
            handle.write(json.dumps({'id': identity}) + '\n')
    conn.execute('DELETE FROM examples WHERE prompt IN (SELECT prompt FROM conflicts)')
    # Same normalized prompt can occur in multiple source representations/groups.
    # If any representation is held out, keep ALL representations out of train.
    conn.execute('CREATE INDEX group_idx ON examples(grp)')
    while True:
        before = conn.total_changes
        conn.execute("UPDATE examples SET split='validation' WHERE split='train' AND prompt IN (SELECT prompt FROM examples WHERE split='validation')")
        conn.execute("UPDATE examples SET split='validation' WHERE split='train' AND grp IN (SELECT grp FROM examples WHERE split='validation')")
        if conn.total_changes == before:
            break
    conn.execute('CREATE INDEX selection_idx ON examples(domain,split,stratum,priority)')
    conn.execute('CREATE TABLE selected (id TEXT PRIMARY KEY, split TEXT)')
    available_totals = {domain: 0 for domain in ('legal', 'general')}
    available_totals.update(dict(conn.execute("SELECT domain,COUNT(*) FROM examples WHERE split='train' GROUP BY domain")))
    write_json(out / 'quota_report.json', {
        'available_train': available_totals,
        'requested': {'legal': args.legal_examples, 'general': args.general_examples},
        'policy': 'all eligible legal if omitted; match general; 50:50 without replacement',
    })
    conn.commit()
    targets = resolve_training_targets(args.legal_examples, args.general_examples, available_totals)
    quotas = {}
    for domain, target in targets.items():
        for split in ('train', 'validation'):
            available = dict(conn.execute('SELECT stratum,COUNT(*) FROM examples WHERE domain=? AND split=? GROUP BY stratum', (domain, split)))
            desired = target if split == 'train' else min(args.validation_examples_per_domain, sum(available.values()))
            if target == 0:
                desired = 0
            quota = allocate(available, desired)
            quotas[domain + '/' + split] = {'available': available, 'selected': quota}
            for stratum, number in quota.items():
                conn.execute('INSERT INTO selected SELECT id,split FROM examples WHERE domain=? AND split=? AND stratum=? ORDER BY priority,id LIMIT ?', (domain, split, stratum, number))
    conn.commit()
    features = Features({key: Sequence(Value('int32')) for key in ('input_ids', 'attention_mask', 'labels')})
    stats = {}
    result = {}
    # Persist canonical conversations plus separate tokenized Arrow data.
    for split in ('train', 'validation'):
        token_counts = Counter()
        def generate():
            with (out / (split + '.jsonl')).open('w', encoding='utf-8') as canonical:
                query = 'SELECT e.id,e.domain,e.payload,e.tokens FROM examples e JOIN selected s ON e.id=s.id WHERE s.split=? ORDER BY e.priority,e.id'
                for identity, domain, payload, tokens in conn.execute(query, (split,)):
                    row = json.loads(tokens)
                    record = json.loads(payload)
                    record.update(id=identity, domain=domain)
                    canonical.write(json.dumps(record, ensure_ascii=False) + '\n')
                    token_counts[domain + '/examples'] += 1
                    token_counts[domain + '/tokens'] += len(row['input_ids'])
                    token_counts[domain + '/assistant_tokens'] += sum(x != -100 for x in row['labels'])
                    yield row
        # Explicit fingerprint prevents hashing the live SQLite connection.
        result[split] = Dataset.from_generator(generate, features=features, cache_dir=str(out / 'arrow_cache'), fingerprint=digest(str(out) + split + tokenizer_signature(tokenizer)))
        if not len(result[split]):
            raise ValueError(f'Empty {split} split; increase the source pool or validation fraction')
        stats[split] = dict(token_counts)
    DatasetDict(result).save_to_disk(str(out / 'dataset'))
    manifest = {'version': 1, 'protocol': 'SaulLM-inspired general sources plus English LawInstruct',
                'seed': args.seed, 'max_length': args.max_length, 'validation_fraction': args.validation_fraction,
                'tokenizer_signature': tokenizer_signature(tokenizer), 'template': CHAT_TEMPLATE,
                'counts': dict(counts), 'quotas': quotas, 'statistics': stats, 'inputs': inputs,
                'requested_training_examples': {'legal': args.legal_examples, 'general': args.general_examples},
                'resolved_training_examples': targets,
                'mmlu_inputs': guard.files, 'mmlu_policy': '>=0.8 question 5-gram containment; dev/validation/test',
                'near_deduplication': 'not performed across SFT pool; normalized prompt-response pairs and prompt conflicts only',
                'validation_grouping': 'LawInstruct prompt / MetaMath original_question / conversation user text; exact prompt cross-source holdout',
                'smoke_subset': bool(args.max_records_per_source),
                'canonical_sha256': {split: file_hash(out / (split + '.jsonl')) for split in result}}
    conn.close()
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest['statistics'], indent=2), flush=True)
    return manifest


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        length = math.ceil(max(len(e['input_ids']) for e in examples) / 8) * 8
        batch = {key: [] for key in ('input_ids', 'attention_mask', 'labels')}
        for example in examples:
            padding = length - len(example['input_ids'])
            for key, pad in (('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)):
                batch[key].append(example[key] + [pad] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
