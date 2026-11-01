"""Training entry point; defaults to a config-only dry run."""
import argparse
import json
from pathlib import Path
import sys

from .config import DEFAULT_CONFIG, describe, load_config


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'export': return export_main(argv[1:])
    if argv and argv[0] == 'bootstrap': return bootstrap_main(argv[1:])
    if argv and argv[0] == 'train': argv.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--train', action='store_true', help='Explicitly start training; otherwise only print configuration')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    for name in ('steps', 'group-size', 'max-tokens', 'max-seconds', 'memory-gb', 'seed'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--learning-rate', type=float)
    options = parser.parse_args(argv)
    overrides = {k:v for k,v in vars(options).items() if k not in {'config', 'train'} and v is not None}
    try:
        args = load_config(options.config, overrides)
        args.train = options.train
        if not args.train:
            print(json.dumps(describe(args), indent=2)); return 0
        from .preflight import validate_local_model
        validate_local_model(args)
        if args.method == 'grpo':
            if args.reward_kind == 'tool_episodes':
                from .episode_grpo import run
            else:
                from .grpo import run
        else:
            from .sft import run
        run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


def bootstrap_main(argv=None):
    from .bootstrap import export_demonstrations
    parser = argparse.ArgumentParser(description='Create verified scripted SFT demonstrations; no model calls.')
    parser.add_argument('--output', type=Path, default=Path('data/training/tool-demonstrations'))
    parser.add_argument('--tasks', type=int, default=24)
    parser.add_argument('--seed', type=int, default=3812)
    options = parser.parse_args(argv)
    try:
        report = export_demonstrations(options.output, options.tasks, options.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}))
    return 0


def export_main(argv=None):
    from .dataset import export
    parser = argparse.ArgumentParser(description='Audit local trajectories; no model or training calls.')
    parser.add_argument('roots', nargs='*', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args(argv)
    try:
        args = load_config(options.config)
        root = Path(args.source_config['project_root'])
        roots = [(root / p).resolve() for p in options.roots] if options.roots else args.trajectory_paths
        destination = (root / options.output).resolve() if options.output else args.dataset
        paths = [p for item in roots for p in ([item] if item.is_file() else item.rglob('trajectories/*/events.jsonl'))]
        report = export(paths, destination)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Audited {report['traces']} traces; exported {report['sft_rows']} eligible SFT steps. GRPO needs fresh local rollouts.")
    return 0
