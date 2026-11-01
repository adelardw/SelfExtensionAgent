import argparse
import json
from pathlib import Path

from .audit import audit
from .integrity import enroll


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local security audit; no model calls or active network scanning')
    parser.add_argument('command', choices=('audit', 'enroll'), nargs='?', default='audit')
    parser.add_argument('--network', action='store_true', help='Passive socket metadata for the current OS user')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = enroll() if args.command == 'enroll' else audit(network=args.network)
        text = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text); args.output.chmod(0o600)
        print(text)
    except (OSError, ValueError) as exc: parser.error(str(exc))
    return int(any(f['severity']=='critical' for f in report.get('findings', [])))


if __name__ == '__main__': raise SystemExit(main())
