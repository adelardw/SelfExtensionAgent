"""Strict YAML configs without importing MLX or starting inference."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name('configs') / 'base.yml'
KEYS = {
    '': {'version', 'project_root', 'model', 'train', 'runtime', 'output'},
    'model': {'name', 'path', 'revision', 'mlx', 'max_parameters_b', 'init_adapter'},
    'train': {'type', 'steps', 'seed', 'type_cfg', 'qat', 'lora', 'datasets', 'sft', 'episodes'},
    'train.episodes': {'max_turns'},
    'train.type_cfg': {'reward', 'learning_rate', 'per_device_train_batch_size',
        'gradient_accumulation_steps', 'num_generations', 'max_completion_length',
        'max_prompt_length', 'kl_beta', 'temperature', 'top_p', 'use_vllm', 'bf16'},
    'train.type_cfg.reward': {'kind', 'model_name'},
    'train.lora': {'type', 'cfg'},
    'train.lora.cfg': {'r', 'scale', 'target_modules', 'lora_dropout', 'bias'},
    'train.datasets': {'source', 'os', 'trajectory_paths', 'path', 'heldout_count'},
    'train.sft': {'mask_prompt', 'val_batches', 'steps_per_report', 'steps_per_eval', 'save_every'},
    'runtime': {'memory_gb', 'cache_mb', 'max_seconds', 'gradient_checkpointing'},
    'output': {'path'},
}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def _read(path, chain=()):
    path = Path(path).resolve()
    if path in chain or len(chain) >= 8: raise ValueError('Config inheritance cycle/depth limit')
    try:
        with path.open() as f: data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid YAML in {path}: {exc}') from exc
    if not isinstance(data, dict): raise ValueError('Config must be a YAML mapping')
    parent = data.pop('extends', None)
    if parent is not None and not isinstance(parent, str): raise ValueError('extends must be one config path')
    if 'project_root' in data:
        data['project_root'] = str((path.parent / _path(data['project_root'], 'project_root')).resolve())
    return merge(_read(path.parent / parent, (*chain, path)), data) if parent else data


def _path(value, name):
    if not isinstance(value, str) or not value.strip() or value == '...':
        raise ValueError(name + ' must be a nonempty path')
    return Path(value).expanduser()


def _shape(data, prefix=''):
    if not isinstance(data, dict): raise ValueError((prefix or 'config') + ' must be a mapping')
    unknown = set(data) - KEYS[prefix]
    if unknown: raise ValueError('Unknown config keys: ' + ', '.join(prefix + '.' + str(k) for k in unknown))
    for key, value in data.items():
        path = f'{prefix}.{key}' if prefix else key
        if path in KEYS: _shape(value, path)


def _number(value, name, low, high, integer=False):
    if isinstance(value, bool): raise ValueError(name + ' must be numeric, not boolean')
    try: number = float(value)
    except (ValueError, TypeError): raise ValueError(name + ' must be numeric') from None
    if not math.isfinite(number) or not low <= number <= high or integer and number != int(number):
        raise ValueError(f'{name} must be in [{low}, {high}]' + (' and integral' if integer else ''))
    return int(number) if integer else number


def load_config(path=DEFAULT_CONFIG, overrides=None):
    raw = merge(_read(DEFAULT_CONFIG), _read(path))
    _shape(raw)
    if raw['version'] != 1 or type(raw['version']) is not int: raise ValueError('Unsupported config version')
    model, train, runtime = raw['model'], raw['train'], raw['runtime']
    cfg, lora, data = train['type_cfg'], train['lora']['cfg'], train['datasets']
    if train['type'] not in {'grpo', 'sft'}: raise ValueError('train.type must be grpo or sft')
    if model['mlx'] is not True: raise ValueError('Only the MLX backend is implemented')
    if not isinstance(model['name'], str) or not model['name']: raise ValueError('model.name is required')
    if model['revision'] is not None and not isinstance(model['revision'], str): raise ValueError('model.revision must be a string or null')
    for field, value in [('train.qat', train['qat']), ('use_vllm', cfg['use_vllm']), ('bf16', cfg['bf16'])]:
        if value is not False: raise ValueError(field + ' is unsupported; use false for MLX QLoRA')
    reward_kind = cfg['reward']['kind']
    if reward_kind not in {'verified_edits', 'tool_episodes'} or cfg['reward']['model_name'] is not None:
        raise ValueError('Use an external verified_edits or tool_episodes reward; reward models are unsupported')
    if train['lora']['type'] != 'qlora' or lora['bias'] is not None:
        raise ValueError('Use qlora with bias: null')
    targets = lora['target_modules']
    if targets is not None and (not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t.strip() for t in targets)):
        raise ValueError('target_modules must be null or a nonempty list of module suffixes')
    if data['os'] != []: raise ValueError('OS curriculum is not connected; datasets.os must be []')
    expected = reward_kind if train['type'] == 'grpo' else 'trajectories'
    if data['source'] != expected: raise ValueError(f'{train["type"]} requires datasets.source: {expected}')
    if not isinstance(data['trajectory_paths'], list): raise ValueError('trajectory_paths must be a list')
    root = _path(raw['project_root'], 'project_root')
    resolve = lambda value, name: (root / _path(value, name)).resolve()
    args = argparse.Namespace(
        train=False, method=train['type'], reward_kind=reward_kind, config_path=str(Path(path).resolve()),
        model_name=model['name'], model=resolve(model['path'], 'model.path'), model_revision=model['revision'],
        init_adapter=resolve(model['init_adapter'], 'model.init_adapter') if model['init_adapter'] is not None else None,
        output=resolve(raw['output']['path'], 'output.path'),
        dataset=resolve(data['path'], 'datasets.path'),
        trajectory_paths=[resolve(p, 'trajectory_paths') for p in data['trajectory_paths']],
        gradient_checkpointing=runtime['gradient_checkpointing'], target_modules=targets,
        sft=copy.deepcopy(train['sft']), source_config=raw)
    fields = {
        'steps': (train['steps'], 1, 500, True), 'seed': (train['seed'], 0, 2**31-1, True),
        'group_size': (cfg['num_generations'], 2, 8, True),
        'max_tokens': (cfg['max_completion_length'], 64, 512, True),
        'max_prompt_tokens': (cfg['max_prompt_length'], 64, 8192, True),
        'max_seconds': (runtime['max_seconds'], 30, 14400, True),
        'memory_gb': (runtime['memory_gb'], 2, 16, True), 'cache_mb': (runtime['cache_mb'], 0, 2048, True),
        'learning_rate': (cfg['learning_rate'], 1e-9, 1e-3, False),
        'kl_beta': (cfg['kl_beta'], 0, 10, False), 'temperature': (cfg['temperature'], 0.01, 2, False),
        'top_p': (cfg['top_p'], 0.01, 1, False), 'rank': (lora['r'], 1, 64, True),
        'scale': (lora['scale'], 0.01, 128, False), 'dropout': (lora['lora_dropout'], 0, 0.5, False),
        'micro_batch_size': (cfg['per_device_train_batch_size'], 1, 8, True),
        'gradient_accumulation_steps': (cfg['gradient_accumulation_steps'], 1, 64, True),
        'heldout_count': (data['heldout_count'], 1, 31, True),
        'episode_max_turns': (train['episodes']['max_turns'], 2, 12, True),
        'max_parameters_b': (model['max_parameters_b'], 0.1, 10, False),
    }
    overrides = overrides or {}
    unknown = set(overrides) - set(fields) - {'model', 'output'}
    if unknown: raise ValueError('Unknown overrides: ' + ', '.join(sorted(unknown)))
    for key, (value, low, high, integer) in fields.items():
        setattr(args, key, _number(overrides.get(key, value), key, low, high, integer))
    for key in ('model', 'output'):
        if key in overrides: setattr(args, key, resolve(str(overrides[key]), key))
    if type(args.gradient_checkpointing) is not bool: raise ValueError('gradient_checkpointing must be boolean')
    if args.init_adapter is not None and (args.method != 'grpo' or args.reward_kind != 'tool_episodes'):
        raise ValueError('init_adapter is supported only for tool_episodes GRPO')
    if args.method == 'grpo':
        if args.gradient_accumulation_steps != 1: raise ValueError('GRPO accumulates one group internally; gradient_accumulation_steps must be 1')
        if args.micro_batch_size > args.group_size: raise ValueError('GRPO microbatch cannot exceed group size')
        if args.dropout != 0: raise ValueError('GRPO dropout must be zero for deterministic behavior logprobs')
        if args.reward_kind == 'tool_episodes' and (args.temperature != 1 or args.top_p != 1):
            raise ValueError('tool_episodes uses temperature=1 and top_p=1 so rollout logprobs match the sampling policy')
    if args.sft['mask_prompt'] is not True: raise ValueError('SFT mask_prompt must be true to exclude tool/input tokens from loss')
    for key in ('val_batches', 'steps_per_report', 'steps_per_eval', 'save_every'):
        args.sft[key] = _number(args.sft[key], 'sft.'+key, 1, 10000, True)
    return args


def describe(args):
    def serial(value):
        if isinstance(value, Path): return str(value)
        if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
        if isinstance(value, list): return [serial(v) for v in value]
        return value
    return serial(vars(args))
