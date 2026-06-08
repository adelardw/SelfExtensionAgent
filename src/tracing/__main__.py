"""
  python -m src.tracing            # самодиагностика по трейсам за сутки
  python -m src.tracing --hours 72
"""
import argparse

from omegaconf import OmegaConf

from ..memory import MemoryStore, build_embedder
from . import diagnose, trace_store


def main() -> None:
    ap = argparse.ArgumentParser(description="Самодиагностика агента по трейсам")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--user", default="local")
    args = ap.parse_args()

    cfg = OmegaConf.load("config.yml")
    store = MemoryStore(
        db_path=cfg.get("memory", {}).get("db_path", "data/memory.db"),
        embedder=build_embedder(False),
    )
    report = diagnose(store, user_id=args.user, since_hours=args.hours)
    print("Здоров:" if report["healthy"] else "Найдены проблемы:")
    for f in report["findings"]:
        print("  •", f)
    print("\nСтатистика нод (avg ms):")
    for s in report["node_stats"]:
        print(f"  {s['node']:<16} calls={s['calls']:<4} avg={s['avg_ms']:.0f}ms max={s['max_ms']:.0f}ms err={s['errors']}")
    print(f"\nРотация трейсов: удалено {trace_store.prune()} старых спанов.")


if __name__ == "__main__":
    main()
