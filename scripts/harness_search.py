"""Offline harness inventory and bounded Pareto configuration search.

No model calls for plan/audit. run requires --live and an explicit plan; it uses
the existing OpenRouter backend and never extracts Codex credentials.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.harness_search import SearchPlan, campaign, digest, inventory, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--reports', default='data/eval')
    audit.add_argument('--output', default='data/eval/pareto-inventory.json')
    for name in ('plan', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--config', default='experiments/pareto_pilot.json')
        if name == 'run':
            command.add_argument('--live', action='store_true')
            command.add_argument('--output', required=True, help='New campaign directory; never overwritten')
    args = parser.parse_args(argv)
    if args.command == 'audit':
        result = inventory(args.reports)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, result)
        print(f"{len(result['groups'])} attributable groups, {len(result['excluded'])} excluded reports; {args.output}")
        return 0
    plan = SearchPlan.model_validate_json(Path(args.config).read_text())
    configs = plan.configurations()
    if args.command == 'plan' or not args.live:
        import json
        print(json.dumps({'plan': plan.model_dump(), 'effective_knobs': configs,
                          'envelope': plan.envelope(), 'live': False}, ensure_ascii=False, indent=2))
        return 0
    from scripts import quality_ladder
    from omegaconf import OmegaConf
    from src.config.config_paths import load_base_config
    project = Path(__file__).resolve().parents[1]

    async def evaluate(plan, stage, knobs, target, work):
        options = quality_ladder.build_parser().parse_args([])
        for key, value in knobs.items():
            setattr(options, key, value)
        options.model, options.provider, options.reasoning = plan.model, plan.provider, plan.reasoning
        options.seed, options.case, options.suite = stage.seed, stage.cases, 'all'
        options.budget_usd, options.timeout = plan.budget_per_task_usd, plan.timeout_seconds
        options.output, options.work_dir = str(target), str(work)
        options.answer_verification = True
        options.provider_retries = 0
        options.jobs = 1
        await quality_ladder.run(options)

    def fingerprint():
        return {'source': digest(quality_ladder.source_manifest(project)),
                'runner': digest(Path(__file__).read_text()),
                'base_config': digest(OmegaConf.to_container(load_base_config()['coding'], resolve=True))}

    try:
        result = asyncio.run(campaign(plan, args.output, evaluate, fingerprint))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Search stopped; evidence retained: {exc}\n')
    print(f"{result['status']}; selected={result['selected']}; reserved=${result['reserved_usd']:.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
