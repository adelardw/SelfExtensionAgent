"""On-policy, multi-action GRPO with outcome-only rewards and action-only loss."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

from .episodes import ActionSample, ContextLimit, collect_episode, episode_credit, project_tasks


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o600)


def build_batch(model, episodes, pad, beta, checkpoint):
    """Each row keeps the EXACT prompt used for that action; tool text is masked.

Normalize returns over episodes, not over a variable number of tool steps.
Reference passes run one action at a time to bound peak logits memory. The loss
uses a token mean: longer episodes have more action tokens, no length reward.
"""
    import mlx.core as mx
    from mlxrl.policy.logprobs import adapters_disabled, completion_logprobs
    from mlxrl.train import GRPOBatch

    credit = episode_credit(episodes)
    samples = [step['sample'] for _, step, _ in credit]
    prompts = tuple(tuple(s['prompt_tokens']) for s in samples)
    completions = tuple(tuple(s['completion_tokens']) for s in samples)
    width = max(map(len, completions))
    old = mx.array([list(s['old_policy_logprobs']) + [0.] * (width-len(s['completion_tokens'])) for s in samples])
    mask = mx.array([[1.] * len(c) + [0.] * (width-len(c)) for c in completions])
    if beta:
        reference = []
        model.eval()
        with adapters_disabled(model):
            for prompt, completion in zip(prompts, completions):
                result = completion_logprobs(model, [prompt], [completion], pad, use_checkpoint=checkpoint)
                mx.eval(result.logprobs)
                reference.append(result.logprobs[0].tolist() + [0.] * (width-len(completion)))
        reference = mx.array(reference)
    else:
        reference = old
    return GRPOBatch(prompt_token_ids=prompts, completion_token_ids=completions,
        rewards=mx.array([episodes[i]['reward'] for i, _, _ in credit]),
        advantages=mx.array([a for _, _, a in credit]), old_policy_logprobs=mx.stop_gradient(old),
        reference_logprobs=mx.stop_gradient(reference), mask=mask, reference_is_policy=not beta)


def run(args):
    from .config import describe
    if args.output.exists(): raise ValueError('Use a fresh output directory; existing adapters are never overwritten')
    args.output.mkdir(parents=True, mode=0o700)
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlxrl.algo import GRPOAlgorithm
    from mlxrl.policy import LoRAConfig, load_policy_with_lora
    from mlxrl.policy.logprobs import pad_token_id_from_tokenizer
    from mlxrl.rollout.naive import SamplingConfig, generate_completion, eos_token_ids_from_tokenizer
    from mlxrl.train import optimizer_step
    from importlib.metadata import version

    mx.set_memory_limit(args.memory_gb * 1024**3)
    mx.set_cache_limit(args.cache_mb * 1024**2)
    mx.random.seed(args.seed)
    started = time.monotonic()
    model, tokenizer, report = load_policy_with_lora(str(args.model), LoRAConfig(
        rank=args.rank, scale=args.scale, dropout=args.dropout,
        target_suffixes=tuple(args.target_modules) if args.target_modules else None,
        grad_checkpoint=args.gradient_checkpointing))
    if report.total_params > args.max_parameters_b * 1e9 or not report.quantized_linear_count:
        raise ValueError('Model exceeds parameter limit or is not quantized')
    adapter_manifest = None
    if args.init_adapter is not None:
        info = json.loads((args.init_adapter/'adapter_config.json').read_text())
        cfg = info.get('lora_parameters', {})
        if cfg.get('rank') != args.rank or cfg.get('scale') != args.scale or cfg.get('dropout', 0) != args.dropout:
            raise ValueError('Initial adapter rank/scale/dropout must match the GRPO configuration')
        weights_path = args.init_adapter/'adapters.safetensors'
        weights = mx.load(str(weights_path))
        expected = dict(tree_flatten(model.trainable_parameters()))
        if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in expected):
            raise ValueError('Initial adapter tensors must match this model and target modules exactly')
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        adapter_manifest = {'path': str(args.init_adapter), 'sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                            'kl_reference': 'original frozen base without adapters'}
    train = project_tasks(args.seed, args.steps, 'train')
    heldout = project_tasks(args.seed + 1, args.heldout_count, 'test')
    manifest = describe(args) | {'model_report': asdict(report), 'initial_adapter': adapter_manifest,
        'algorithm': 'episode-normalized outcome GRPO; one unclipped update per fresh group; action-token mean loss',
        'scope': 'file search/read/edit/public check/finish; bounded arithmetic Python, not full SWE',
        'reward': '1 only for accepted finish plus external full-domain correctness; 0 otherwise; no shaping',
        'train_families': sorted({t.problem.family for t in train}),
        'heldout_families': sorted({t.problem.family for t in heldout}),
        'versions': {p: version(p) for p in ('mlx', 'mlx-lm', 'inductive-mlxrl')},
        'code': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    write_json(args.output/'manifest.json', manifest)
    pad = int(pad_token_id_from_tokenizer(tokenizer))
    eos = eos_token_ids_from_tokenizer(tokenizer)
    optimizer, algorithm = optim.Adam(learning_rate=args.learning_rate), GRPOAlgorithm()
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    def abort(): return interrupted or time.monotonic()-started >= args.max_seconds
    def adapter_hash():
        h = hashlib.sha256()
        for name, array in tree_flatten(model.trainable_parameters()):
            h.update(name.encode()); h.update(np.array(array).tobytes())
        return h.hexdigest()
    def save_adapter():
        mx.save_safetensors(str(args.output/'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
        write_json(args.output/'adapter_config.json', {'fine_tune_type': 'lora', 'num_layers': report.layer_count,
            'lora_parameters': {'rank': args.rank, 'scale': args.scale, 'dropout': args.dropout,
                               'keys': list(report.lora_target_keys)}})
    def generator(temperature):
        def generate(messages):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt = tuple(tokenizer.encode(rendered))
            if len(prompt) > args.max_prompt_tokens: raise ContextLimit()
            tokens, logprobs, _ = generate_completion(model, tokenizer, prompt,
                SamplingConfig(max_tokens=args.max_tokens, temperature=temperature, top_p=1), eos_token_ids=eos)
            visible = list(tokens)
            while visible and visible[-1] in eos: visible.pop()
            return ActionSample(prompt, tokens, logprobs, tokenizer.decode(visible))
        return generate
    def rollout(task, folder, temperature, seed):
        model.eval(); mx.random.seed(seed)
        result = collect_episode(task, generator(temperature), folder/'project', args.episode_max_turns, abort)
        write_json(folder/'episode.json', result)
        return result
    def evaluate(label):
        results = []
        for i, task in enumerate(heldout):
            if abort(): break
            results.append(rollout(task, args.output/label/str(i), 0, args.seed))
        write_json(args.output/(label+'.json'), results)
        return results

    previous = signal.signal(signal.SIGTERM, stop)
    initial_hash = adapter_hash()
    before, after, metrics, updates = [], [], [], 0
    error = None
    try:
        before = evaluate('before')
        for group, task in enumerate(train):
            if abort(): break
            policy_hash = adapter_hash()
            episodes = []
            for candidate in range(args.group_size):
                if abort(): break
                episode = rollout(task, args.output/'episodes'/str(group)/str(candidate), 1,
                    args.seed + 1000 + group*args.group_size + candidate)
                episodes.append(episode)
            row = {'group': group, 'task_id': task.id, 'policy_revision': updates, 'adapter_sha256': policy_hash,
                   'rewards': [e['reward'] for e in episodes], 'turns': [len(e['steps']) for e in episodes],
                   'updated': False}
            eligible = len(episodes) == args.group_size and all(e['training_eligible'] for e in episodes)
            if eligible and not abort() and len(set(row['rewards'])) > 1:
                batch = build_batch(model, episodes, pad, args.kl_beta, args.gradient_checkpointing)
                mx.savez(str(args.output/f'group-{group}-behavior.npz'), old_policy_logprobs=batch.old_policy_logprobs,
                    reference_logprobs=batch.reference_logprobs, mask=batch.mask, advantages=batch.advantages)
                result = optimizer_step(model, optimizer, batch, beta=args.kl_beta, pad_token_id=pad,
                    algorithm=algorithm, micro_batch_size=args.micro_batch_size, use_checkpoint=args.gradient_checkpointing)
                update_metrics = asdict(result)
                update_metrics['action_mean_reward'] = update_metrics.pop('mean_reward')
                updates += 1; row.update(update_metrics, updated=True)
                save_adapter()
                del batch
            else:
                row['skip_reason'] = 'incomplete_or_interrupted_group' if not eligible or abort() else 'equal_episode_rewards'
            known_rewards = [r for r in row['rewards'] if r is not None]
            row['episode_mean_reward'] = sum(known_rewards)/len(known_rewards) if known_rewards else None
            row.update(seconds=round(time.monotonic()-started, 2), peak_memory_gb=round(mx.get_peak_memory()/1024**3, 3))
            metrics.append(row)
            write_json(args.output/'metrics.json', metrics)
            print(json.dumps(row), flush=True)
        after = evaluate('after')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            save_adapter()
            summary = {'groups': len(metrics), 'optimizer_updates': updates,
                'adapter_changed': initial_hash != adapter_hash(),
                'before': {'solved': sum(e['reward'] == 1 for e in before), 'completed': sum(e['reward'] is not None for e in before), 'expected': len(heldout)},
                'after': {'solved': sum(e['reward'] == 1 for e in after), 'completed': sum(e['reward'] is not None for e in after), 'expected': len(heldout)},
                'seconds': round(time.monotonic()-started, 2), 'peak_memory_gb': round(mx.get_peak_memory()/1024**3, 3),
                'interrupted': interrupted, 'deadline_reached': time.monotonic()-started >= args.max_seconds,
                'error': error, 'quality_gain_established': False,
                'limitation': 'Small arithmetic tool curriculum; no full SWE, benchmark or general improvement claim.'}
            write_json(args.output/'summary.json', summary)
            print(json.dumps(summary), flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous)
