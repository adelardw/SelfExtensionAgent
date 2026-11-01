"""Bounded on-policy GRPO/QLoRA pilot for verified arithmetic code edits on MLX.

This trains a single-action policy, not the full multi-agent harness. Cloud traces
are diagnostic/SFT inputs, never relabeled as on-policy rollouts. No generated code
is executed. Without --train, this command only validates and describes the setup.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

from .verified_edits import task_set


def run(args):
    from .config import describe
    train = task_set(args.seed, max(args.steps, 6), 'train')
    heldout = task_set(args.seed + 1, args.heldout_count, 'test')
    settings = describe(args) | {'model': str(args.model), 'output': str(args.output),
        'algorithm': 'mlxrl GRPO: one unclipped optimizer update per fresh group; KL to frozen base',
        'scope': 'single arithmetic edit, exhaustive integer input domain [-20,20]; not full SWE',
        'train_families': sorted({t.family for t in train}), 'heldout_families': ['distance'],
        'sampling': {'temperature': args.temperature, 'top_p': args.top_p}, 'lora_rank': args.rank}
    if not args.train:
        print(json.dumps(settings, indent=2)); return
    if not args.model.is_dir(): raise ValueError('Download a model first; --model must be a local directory')
    if args.output.exists(): raise ValueError('Use a fresh output directory; existing adapters are never overwritten')
    args.output.mkdir(parents=True, mode=0o700)
    # Optional dependencies stay in the isolated .venv-rl environment.
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlxrl.algo import GRPOAlgorithm
    from mlxrl.policy import LoRAConfig, load_policy_with_lora
    from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
    from mlxrl.rollout.naive import SamplingConfig, generate_group_rollouts
    from mlxrl.train import batch_from_rollouts, optimizer_step
    from importlib.metadata import version

    mx.set_memory_limit(args.memory_gb * 1024**3)
    mx.set_cache_limit(args.cache_mb * 1024**2)
    mx.random.seed(args.seed)
    started = time.monotonic()
    model, tokenizer, model_report = load_policy_with_lora(str(args.model), LoRAConfig(
        rank=args.rank, scale=args.scale, dropout=args.dropout,
        target_suffixes=tuple(args.target_modules) if args.target_modules else None,
        grad_checkpoint=args.gradient_checkpointing))
    if model_report.total_params > args.max_parameters_b * 1e9:
        raise ValueError('Model exceeds configured parameter limit')
    if not model_report.quantized_linear_count:
        raise ValueError('Pilot expects quantized base weights')
    settings['model_report'] = asdict(model_report)
    settings['versions'] = {p: version(p) for p in ('mlx', 'mlx-lm', 'inductive-mlxrl')}
    metadata = args.model / 'download_manifest.json'
    settings['model_revision'] = json.loads(metadata.read_text()) if metadata.exists() else None
    settings['code'] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(__file__).with_name('verified_edits.py'), Path(__file__).with_name('config.py'))}
    (args.output / 'manifest.json').write_text(json.dumps(settings, indent=2) + '\n')

    def adapter_hash():
        h = hashlib.sha256()
        for name, array in tree_flatten(model.trainable_parameters()):
            h.update(name.encode()); h.update(np.array(array).tobytes())
        return h.hexdigest()

    def render(task):
        prompt = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a code-editing API. Return exactly one JSON object. No Markdown fences, commentary or explanations. The first character must be { and the last must be }.'},
            {'role': 'user', 'content': task.prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer.encode(prompt)) > args.max_prompt_tokens:
            raise ValueError('Prompt exceeds max_prompt_length; refusing to truncate the task')
        return prompt

    def text(completion):
        # MLXRL includes terminal EOS in decoded text. APIs normally do not expose
        # it as content. Remove terminal control TOKENS only; never repair JSON.
        from mlxrl.rollout.naive import eos_token_ids_from_tokenizer
        ids = list(completion.completion_tokens)
        eos = eos_token_ids_from_tokenizer(tokenizer)
        while ids and ids[-1] in eos: ids.pop()
        return tokenizer.decode(ids)

    def evaluate(tasks):
        rows = []
        for task in tasks:
            completion = generate_group_rollouts(model, tokenizer, [render(task)], group_size=1,
                config=SamplingConfig(max_tokens=args.max_tokens, temperature=0),
                seed=args.seed, use_chat_template=False)[0]
            rows.append({'task': asdict(task), 'completion': text(completion), 'reward': task.verify(text(completion))})
        return rows

    def save_adapter():
        mx.save_safetensors(str(args.output / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
        (args.output / 'adapter_config.json').write_text(json.dumps({
            'fine_tune_type': 'lora', 'num_layers': model_report.layer_count,
            'lora_parameters': {'rank': args.rank, 'scale': args.scale, 'dropout': args.dropout,
                                'keys': list(model_report.lora_target_keys)}}, indent=2) + '\n')

    initial_hash = adapter_hash()
    before = evaluate(heldout)
    (args.output / 'before.json').write_text(json.dumps(before, indent=2) + '\n')
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    algorithm = GRPOAlgorithm()
    pad = int(pad_token_id_from_tokenizer(tokenizer))
    sampling = SamplingConfig(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
    updates = 0
    metrics = []
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        for step, task in enumerate(train[:args.steps]):
            if interrupted or time.monotonic() - started > args.max_seconds: break
            model.eval()
            policy_hash = adapter_hash()
            completions = generate_group_rollouts(model, tokenizer, [render(task)], group_size=args.group_size,
                config=sampling, seed=args.seed + step, use_chat_template=False)
            rewards = [task.verify(text(c)) for c in completions]
            record = {'group': step, 'policy_revision': updates, 'adapter_sha256': policy_hash,
                'task': asdict(task), 'prompt': render(task), 'rewards': rewards,
                'completions': [asdict(c) for c in completions],
                'verifier_text': [text(c) for c in completions],
                'verifier': 'verified_edits.py:' + settings['code']['verified_edits.py']}
            with (args.output / 'rollouts.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
            row = {'group': step, 'rewards': rewards, 'updated': False}
            # A tied group carries no relative task signal; never invent a winner.
            if len(set(rewards)) > 1 and all(c.completion_tokens for c in completions):
                batch = batch_from_rollouts(model, completions, rewards, args.group_size, pad, algorithm,
                    compute_reference=args.kl_beta != 0, use_checkpoint=args.gradient_checkpointing)
                # Freeze actual old-policy logprobs/advantages, with the generated token IDs.
                mx.savez(str(args.output / f'group-{step}-behavior.npz'),
                    old_policy_logprobs=batch.old_policy_logprobs, mask=batch.mask,
                    advantages=batch.advantages, reference_logprobs=batch.reference_logprobs)
                result = optimizer_step(model, optimizer, batch, beta=args.kl_beta, pad_token_id=pad,
                    algorithm=algorithm, micro_batch_size=args.micro_batch_size, use_checkpoint=args.gradient_checkpointing)
                updates += 1
                row.update(asdict(result), updated=True)
                save_adapter()
            row['seconds'] = round(time.monotonic() - started, 2)
            row['peak_memory_gb'] = round(mx.get_peak_memory() / 1024**3, 3)
            metrics.append(row)
            with (args.output / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        model.eval()
        after = evaluate(heldout) if not interrupted and time.monotonic()-started < args.max_seconds else []
        save_adapter()
        final_hash = adapter_hash()
        summary = {'groups': len(metrics), 'optimizer_updates': updates,
            'adapter_changed': initial_hash != final_hash, 'initial_adapter_sha256': initial_hash,
            'final_adapter_sha256': final_hash, 'before': before, 'after': after,
            'seconds': round(time.monotonic() - started, 2),
            'peak_memory_gb': round(mx.get_peak_memory() / 1024**3, 3),
            'interrupted': interrupted, 'quality_gain_established': False,
            'limitation': 'Synthetic held-out-family cases only; no full-agent or benchmark quality claim.'}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k:v for k,v in summary.items() if k not in {'before','after'}}), flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
