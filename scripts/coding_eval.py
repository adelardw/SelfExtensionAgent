"""Small portable coding evaluation. Default lists tasks without any model calls.

Run: uv run python scripts/coding_eval.py --live --backend openrouter --reasoning low
Uses disposable workspaces, independent unittest verification and provider usage.
This is a smoke benchmark, not a claim about general software-engineering ability.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
import uuid
import hashlib
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = [
    {"id": "repair", "task": "Fix average: ignore None, return 0 for an empty input. Preserve the API; run checks.",
     "files": {"app.py": "def average(values):\n    return sum(values) / len(values)\n",
               "test_app.py": "import unittest\nfrom app import average\nclass Test(unittest.TestCase):\n def test_average(self):\n  self.assertEqual(average([2, None, 4]), 3)\n  self.assertEqual(average([]), 0)\n  self.assertEqual(average([None]), 0)\n"}},
    {"id": "feature", "task": "Add parse_port(value) to config.py and use it from address(). Accept integers or digit strings 1..65535; reject booleans and invalid values with ValueError. Verify.",
     "files": {"config.py": "def address(host, port):\n    return f'{host}:{port}'\n",
               "test_config.py": "import unittest\nimport config\nclass Test(unittest.TestCase):\n def test_address(self):\n  self.assertEqual(config.address('localhost', '8080'), 'localhost:8080')\n def test_port(self):\n  for p in ['443', 80, 65535]:\n   self.assertEqual(config.parse_port(p), int(p))\n  for p in [True, False, 0, 65536, 'abc', -1, 1.2]:\n   with self.assertRaises(ValueError): config.parse_port(p)\n"}},
]


async def evaluate(opts):
    from src.coding.harness import CodingHarness
    from src.config.config_paths import load_base_config
    from src.runtime import hitl, run_context
    check = [sys.executable, "-B", "-m", "unittest", "-q"]
    results = []
    source = Path(__file__).resolve().parents[1]/'src'/'coding'
    harness_id = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source.glob('*.py')))).hexdigest()[:16]
    for case in CASES:
        if opts.work_dir:
            workspace = Path(opts.work_dir) / (case["id"] + "-" + uuid.uuid4().hex[:8])
            workspace.mkdir(parents=True)
            location = nullcontext(str(workspace))
        else:
            location = tempfile.TemporaryDirectory(prefix="sea-code-eval-")
        with location as temp:
            root = Path(temp)
            for name, content in case["files"].items():
                (root / name).write_text(content)
            config = dict(load_base_config().get("coding", {}))
            config.update(model=opts.model, backend=opts.backend, base_url=opts.base_url,
                          max_turns=opts.max_turns, checks={"unit": check}, allow_web=False, learn=False,
                          supervisor_review=opts.supervisor_review)
            started = time.monotonic()
            with run_context.request_scope("eval-" + case["id"], "coding-eval"):
                hitl.set_work_mode("auto")
                agent = CodingHarness(root, config=config)
                try:
                    result = await agent.run(case["task"], reasoning=opts.reasoning)
                except Exception as exc:
                    result = {"status": "error", "error": str(exc), **agent.model.usage()}
            # Restore the original assertions before independent validation: weakening
            # repository tests must not manufacture a benchmark success.
            for name, content in case["files"].items():
                if name.startswith("test_"):
                    (root / name).write_text(content)
            verified = subprocess.run(check, cwd=root, capture_output=True, text=True, timeout=30)
            row = {"task": case["id"], "model": opts.model, "status": result["status"],
                   'evaluation':'portable_smoke_not_terminalbench', 'harness_id':harness_id,
                   "passed": verified.returncode == 0, "seconds": round(time.monotonic()-started, 2),
                   "backend": opts.backend,
                   **{k: result.get(k) for k in ("cost", "decision_usage", "tokens_in", "tokens_out", "calls", "delegates", "error", "cached_tokens", "generation_tokens_per_second", "model_file", "parameter_count", "model_size_bytes", "reasoning_calls")}}
            if opts.work_dir:
                row["workspace"] = str(root)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    if opts.output:
        Path(opts.output).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    return 0 if all(r["passed"] and r["status"] == "done" for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--backend", choices=["ollama", "llama_cpp", "openrouter"], default="openrouter")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--reasoning", choices=["auto", "low", "medium", "high"], default="auto")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--output")
    parser.add_argument("--work-dir", help="retain traces and changed files in fresh subdirectories")
    parser.add_argument('--supervisor-review', action='store_true', help='ablation: bounded model-based anomaly review')
    opts = parser.parse_args()
    if not opts.live:
        for case in CASES:
            print(case["id"] + ": " + case["task"])
        print("No model calls made. Pass --live with the configured backend to measure quality/tokens/time.")
        return 0
    return asyncio.run(evaluate(opts))


if __name__ == "__main__":
    raise SystemExit(main())
