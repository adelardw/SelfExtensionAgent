"""Serve a downloaded Unsloth GGUF locally with llama.cpp/Metal.

python scripts/serve_gguf.py --model-file data/models/unsloth/Qwen3.5-4B-UD-Q4_K_XL.gguf
"""
import argparse
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--alias", default="sea-local")
    opts = parser.parse_args()
    executable = shutil.which("llama-server")
    if not executable:
        parser.error("llama-server not installed (macOS: brew install llama.cpp)")
    path = opts.model_file.resolve(strict=True)
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            parser.error("not a GGUF model")
    args = [executable, "--model", str(path), "--alias", opts.alias,
            "--host", "127.0.0.1", "--port", str(opts.port),
            "--ctx-size", str(opts.context), "--parallel", "1", "--n-gpu-layers", "99",
            "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--batch-size", "512", "--ubatch-size", "128", "--threads", "8",
            "--jinja", "--reasoning-format", "deepseek", "--reasoning-budget", "1536"]
    print("Serving", path.name, "at", f"http://127.0.0.1:{opts.port}/v1", flush=True)
    os.execv(executable, args)


if __name__ == "__main__":
    main()
