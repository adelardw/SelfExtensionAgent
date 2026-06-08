"""
  python -m src.maintenance            # обновить зависимости (с откатом при поломке)
  python -m src.maintenance --dry-run  # показать, что обновилось бы
"""
import argparse

from .dep_update import run_update


def main() -> None:
    ap = argparse.ArgumentParser(description="Безопасное авто-обновление зависимостей")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(run_update(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
