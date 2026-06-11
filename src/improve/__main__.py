"""
CLI для self-learning пайпа.

  python -m src.improve              # запустить оптимизацию execution-промпта
  python -m src.improve --role execution --min-failures 3
  python -m src.improve --list       # показать принятые overrides
  python -m src.improve --revert execution
"""
import argparse

from omegaconf import OmegaConf

from ..memory import MemoryStore, build_embedder
from . import SelfLearningPipe, list_overrides, revert


def main() -> None:
    cfg = OmegaConf.load("config.yml")
    ap = argparse.ArgumentParser(description="Self-learning prompt optimization")
    ap.add_argument("--role", default="step_execution")
    ap.add_argument("--min-failures", type=int, default=3)
    ap.add_argument("--graph", action="store_true", help="graph-backward: credit assignment по трейсу + батч-опт")
    ap.add_argument("--user", metavar="USER_ID", help="per-user backward: уроки из неудач юзера → персональные few-shots")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--revert", metavar="ROLE")
    ap.add_argument("--no-accept", action="store_true", help="только предложить, не сохранять")
    args = ap.parse_args()

    if args.list:
        print("Параметры графа:", list_overrides() or "нет")
        return
    if args.revert:
        print("Откат:", "ok" if revert(args.revert) else "не найдено")
        return

    store = MemoryStore(
        db_path=cfg.get("memory", {}).get("db_path", "data/memory.db"),
        embedder=build_embedder(cfg.get("memory", {}).get("embeddings", False)),
    )
    if args.user:
        from . import graph_backward_user

        print(graph_backward_user(store, args.user, min_batch=args.min_failures, accept=not args.no_accept))
    elif args.graph:
        from . import batch_optimize

        print(batch_optimize(store, min_batch=args.min_failures, accept=not args.no_accept))
    else:
        res = SelfLearningPipe(store).run(
            role=args.role, min_failures=args.min_failures, accept=not args.no_accept
        )
        print(res)


if __name__ == "__main__":
    main()
