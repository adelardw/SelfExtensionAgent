"""Рендер реального LangGraph-графа агента в PNG + mermaid (как в LangSmith).

    python scripts/render_graph.py

Пишет docs/agent_graph.png и docs/agent_graph.mmd из build_graph().get_graph().
PNG идёт через mermaid.ink (нужна сеть); mermaid-текст рендерится офлайн.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import build_graph  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> int:
    g = build_graph().get_graph()
    (DOCS / "agent_graph.mmd").write_text(g.draw_mermaid(), encoding="utf-8")
    print(f"mermaid → docs/agent_graph.mmd ({len(g.nodes)} узлов, {len(g.edges)} рёбер)")
    try:
        (DOCS / "agent_graph.png").write_bytes(g.draw_mermaid_png())
        print("PNG → docs/agent_graph.png")
    except Exception as e:  # сети нет / mermaid.ink недоступен — mermaid-текст уже сохранён
        print(f"PNG пропущен ({e}); используйте docs/agent_graph.mmd")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
