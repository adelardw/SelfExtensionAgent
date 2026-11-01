"""Audited chat SFT through MLX-LM; imported GPU dependencies only on execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


def native_config(args):
    lora = {'rank': args.rank, 'scale': args.scale, 'dropout': args.dropout}
    if args.target_modules: lora['keys'] = args.target_modules
    return {'model': str(args.model), 'train': True, 'test': False, 'data': str(args.dataset),
        'adapter_path': str(args.output), 'fine_tune_type': 'lora', 'num_layers': -1,
        'seed': args.seed, 'iters': args.steps, 'learning_rate': args.learning_rate,
        'batch_size': args.micro_batch_size, 'grad_accumulation_steps': args.gradient_accumulation_steps,
        'max_seq_length': args.max_prompt_tokens + args.max_tokens,
        'grad_checkpoint': args.gradient_checkpointing, 'lora_parameters': lora,
        'trust_remote_code': False, 'report_to': [], **args.sft}


def validate_dataset(path):
    path = Path(path)
    seen = set()
    counts = {}
    for split in ('train', 'valid', 'test'):
        split_prompts = set()
        file = path / (split + '.jsonl')
        rows = file.read_text().splitlines() if file.is_file() else []
        counts[split] = 0
        for line in rows:
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row, dict): raise ValueError('SFT rows must be JSON objects')
            messages = row.get('messages')
            if (not isinstance(messages, list) or len(messages) < 2 or
                any(not isinstance(m, dict) or not isinstance(m.get('content'), str) or
                    m.get('role') not in {'system', 'user', 'assistant'} or m.get('tool_calls') for m in messages) or
                messages[-1]['role'] != 'assistant' or not messages[-1]['content'].strip()):
                raise ValueError('SFT expects text chat rows with a final assistant completion')
            if row.get('tools'): raise ValueError('Native tool-template SFT is not connected')
            # Catch identical tasks crossing splits even if their answers differ.
            key = json.dumps(messages[:-1], sort_keys=True, ensure_ascii=False)
            if split != 'train' and key in seen: raise ValueError('Prompt leakage between dataset splits')
            split_prompts.add(key); counts[split] += 1
        seen.update(split_prompts)
    if not counts['train']: raise ValueError('SFT train.jsonl is absent or empty; first collect/export eligible trajectories')
    return counts


def run(args):
    from .config import describe
    counts = validate_dataset(args.dataset)
    if counts['train'] < args.micro_batch_size:
        raise ValueError('SFT dataset is smaller than the configured batch size')
    if args.output.exists(): raise ValueError('Use a fresh output directory; existing adapters are never overwritten')
    args.output.mkdir(parents=True, mode=0o700)
    plan = {'settings': describe(args), 'native': native_config(args), 'dataset_rows': counts}
    job = args.output / 'sft-job.json'
    job.write_text(json.dumps(plan, indent=2) + '\n')
    # MLX lives only in this child. Timeout also covers model loading/backward.
    try:
        with (args.output / 'training.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'src.agent_train.sft', str(job)],
                cwd=Path(__file__).resolve().parents[2], env=os.environ | {'HF_HUB_OFFLINE': '1'},
                stdout=log, stderr=subprocess.STDOUT, timeout=args.max_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('SFT time limit reached; inspect checkpoints and training.log in ' + str(args.output)) from exc
    if result.returncode: raise ValueError('MLX SFT failed; inspect ' + str(args.output / 'training.log'))


def execute(job):
    from types import SimpleNamespace
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import get_total_parameters
    from mlxrl.policy.model import discover_lora_target_keys

    settings, config = job['settings'], copy.deepcopy(job['native'])
    mx.set_memory_limit(settings['memory_gb'] * 1024**3)
    mx.set_cache_limit(settings['cache_mb'] * 1024**2)
    model, tokenizer = load(config['model'], tokenizer_config={'trust_remote_code': False}, trust_remote_code=False)
    if get_total_parameters(model) > settings['max_parameters_b'] * 1e9:
        raise ValueError('Model exceeds configured parameter limit')
    if settings['target_modules']:
        config['lora_parameters']['keys'] = list(discover_lora_target_keys(model, settings['target_modules']))
    native = SimpleNamespace(**(copy.deepcopy(CONFIG_DEFAULTS) | config))
    train_set, valid_set, _ = load_dataset(native, tokenizer)
    train_model(native, model, train_set, valid_set)


if __name__ == '__main__':
    execute(json.loads(Path(sys.argv[1]).read_text()))
