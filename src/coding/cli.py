"""`sea solve` / `sea code`: one evidence-driven loop for software, mathematics and algorithms."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main(args=None):
    parser = argparse.ArgumentParser(prog="sea solve (alias: sea code)")
    parser.add_argument("task", nargs="?", default="")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reasoning", choices=["auto", "low", "medium", "high"], default="auto")
    parser.add_argument("--backend", choices=["llama_cpp", "ollama", "openrouter"])
    parser.add_argument("--base-url", help="local llama-server URL, e.g. http://127.0.0.1:8081/v1")
    parser.add_argument("--model", help="OpenRouter model id, llama-server alias or Ollama model name")
    parser.add_argument('--action-protocol',choices=['json','tools'],help='OpenRouter action transport; both use the same runtime permissions and checks')
    parser.add_argument("--same-model", action="store_true", help="use the selected model for workers, decisions and critique")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--auto", action="store_true", help="authorize project edits and commands")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session", default="cli")
    parser.add_argument("--json", action="store_true")
    parser.add_argument('--interactive', '-i', action='store_true')
    opts = parser.parse_args(args)
    from src.runtime import hitl, run_context
    from .harness import CodingHarness
    from src.config.config_paths import load_base_config
    config = dict(load_base_config().get("coding", {}))
    for name, value in (("backend", opts.backend), ("base_url", opts.base_url), ("model", opts.model),('action_protocol',opts.action_protocol)):
        if value is not None:
            config[name] = value
    if opts.same_model:
        config['orchestration']={**config.get('orchestration',{}),'same_model':True}
    if opts.interactive or (not opts.task and not opts.resume):
        from .terminal import DeveloperConsole
        asyncio.run(DeveloperConsole(opts.root, config, auto=opts.auto, plan=opts.plan,
                                     reasoning=opts.reasoning, session=opts.session).run(opts.task))
        return 0

    async def confirm(text):
        answer = await asyncio.to_thread(input, text + "\nРазрешить? [да/нет] ")
        return answer.strip().lower() in {"да", "yes", "y"}

    async def run():
        with run_context.request_scope("coding-cli", "coding-cli"):
            hitl.set_work_mode("plan" if opts.plan else "auto" if opts.auto else "manual")
            hitl.set_confirmer(confirm)
            return await CodingHarness(opts.root, config=config).run(opts.task, reasoning=opts.reasoning,
                execution="plan" if opts.plan else "auto", session_id=opts.session, resume=opts.resume,
                progress=None if opts.json else lambda e: print(e["message"], flush=True))
    try:
        result = asyncio.run(run())
    except Exception as exc:
        parser.exit(1, f"coding: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2) if opts.json else result["final_answer"])
    return 0 if result["status"] in {"done", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
