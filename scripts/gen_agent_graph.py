"""Генератор схемы РЕАЛЬНОГО графа агента (src/agent.py) → .drawio (→ SVG через drawio_to_svg.py).

Это «граф как граф»: узлы — имена нод LangGraph, без описаний; рёбра — сплошные (add_edge)
и пунктирные (add_conditional_edges), ровно как их строит StateGraph. Раскладка — слоёная
(rank сверху вниз), кросс- и обратные рёбра разведены по боковым полосам, чтобы линии не
шли сквозь узлы. Стиль — лавандовый, как у штатного рендера LangGraph, но крупнее.

    python scripts/gen_agent_graph.py docs/architecture.en.drawio "<title>"
    python scripts/drawio_to_svg.py   docs/architecture.en.drawio docs/architecture.en.svg
"""
from __future__ import annotations

import sys
from html import escape

# --- сетка раскладки (центр узла = (col_x, 90 + rank*ROWH)) ---
ROWH = 128
def Y(r): return 90 + r * ROWH
C1, C2, C3, C4 = 380, 640, 900, 1120          # колонки: пайплайн / ветка / спайн-вход / fast

# имя → (центр_x, rank, тип)   тип: "node" | "io"
NODES = {
    "__start__":           (C3, 0,  "io"),
    "recall":              (C3, 1,  "node"),
    "reflexion":           (C3, 2,  "node"),
    "act":                 (C3, 3,  "node"),
    "goal":                (C1, 3,  "node"),
    "fast_answer":         (C4, 3,  "node"),
    "clarify_gate":        (C2, 4,  "node"),
    "reason":              (C4, 4,  "node"),
    "router":              (C1, 5,  "node"),
    "create_skills":       (C2, 6,  "node"),
    "sgr_create":          (C2, 7,  "node"),
    "skill_selector":      (C1, 8,  "node"),
    "capability_research": (C1, 9,  "node"),
    "decompose":           (C1, 10, "node"),
    "skill_injection":     (C1, 11, "node"),
    "step_executor":       (C1, 12, "node"),
    "synthesize":          (C1, 13, "node"),
    "review":              (C2, 14, "node"),
    "validation":          (C1, 15, "node"),
    "reflect":             (C3, 16, "node"),
    "__end__":             (C3, 17, "io"),
}

# рёбра: (src, tgt, dashed?, waypoints[])  — пустые wp = прямая (клип центр→центр)
SOLID, DASH = False, True
EDGES = [
    # --- сплошные (graph.add_edge) ---
    ("__start__", "recall", SOLID, []),
    ("recall", "reflexion", SOLID, []),
    ("clarify_gate", "router", SOLID, []),
    ("create_skills", "sgr_create", SOLID, []),
    ("skill_selector", "capability_research", SOLID, []),
    ("capability_research", "decompose", SOLID, []),
    ("decompose", "skill_injection", SOLID, []),
    ("skill_injection", "step_executor", SOLID, []),
    ("reason", "validation", SOLID, [(1190, Y(4) + 28), (1190, Y(15))]),
    ("fast_answer", "reflect", SOLID, [(1210, Y(3) + 28), (1210, Y(16))]),
    ("reflect", "__end__", SOLID, []),
    # --- пунктирные (graph.add_conditional_edges) ---
    ("reflexion", "act", DASH, []),
    ("reflexion", "goal", DASH, []),
    ("reflexion", "fast_answer", DASH, []),
    ("act", "goal", DASH, []),
    ("act", "reflect", DASH, []),
    ("goal", "reason", DASH, []),
    ("goal", "clarify_gate", DASH, []),
    ("goal", "router", DASH, []),
    ("router", "create_skills", DASH, []),
    ("router", "skill_selector", DASH, []),
    ("sgr_create", "router", DASH, [(70, Y(7)), (70, Y(5))]),
    ("sgr_create", "create_skills", DASH, [(545, Y(7)), (545, Y(6))]),
    ("sgr_create", "skill_selector", DASH, []),
    ("step_executor", "step_executor", DASH, [(250, Y(12) - 18), (250, Y(12) + 18)]),
    ("synthesize", "review", DASH, []),
    ("synthesize", "validation", DASH, []),
    ("review", "step_executor", DASH, [(98, Y(14)), (98, Y(12))]),
    ("review", "validation", DASH, []),
    ("validation", "router", DASH, [(126, Y(15)), (126, Y(5))]),
    ("validation", "reflect", DASH, []),
]


def box(name, cx, cy, kind):
    if kind == "io":
        w, h = 150, 48
        style = "ellipse;whiteSpace=wrap;html=1;fillColor=#E6E6FA;strokeColor=#9370DB;fontSize=15;fontStyle=1;"
    else:
        w = 210 if len(name) > 16 else 180
        h = 54
        style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ECECFF;strokeColor=#9370DB;fontSize=15;"
    x, y = cx - w / 2, cy - h / 2
    return (f'        <mxCell id="{name}" value="{escape(name)}" style="{style}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w}" height="{h}" as="geometry" />\n'
            f'        </mxCell>')


def edge(i, src, tgt, dashed, wps):
    color = "#8a8fb8" if dashed else "#4a4a6a"
    style = f"html=1;endArrow=block;rounded=1;strokeColor={color};"
    if dashed:
        style += "dashed=1;"
    arr = ""
    if wps:
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in wps)
        arr = f'<Array as="points">{pts}</Array>'
    return (f'        <mxCell id="e{i}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">\n'
            f'          <mxGeometry relative="1" as="geometry">{arr}</mxGeometry>\n'
            f'        </mxCell>')


def main() -> int:
    dst = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "agent graph (LangGraph forward graph)"
    parts = [
        '<mxfile host="app.diagrams.net" agent="self-extension-agent">',
        '  <diagram name="agent-graph" id="agent-graph">',
        '    <mxGraphModel dx="1600" dy="1100" grid="0" gridSize="10" guides="1" tooltips="1" '
        'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1300" pageHeight="2400" '
        'math="0" shadow="0">',
        '      <root>',
        '        <mxCell id="0" />',
        '        <mxCell id="1" parent="0" />',
        f'        <mxCell id="title" value="{escape(title)}" style="text;html=1;fontSize=17;fontStyle=1;align=left;verticalAlign=middle;" vertex="1" parent="1">\n'
        f'          <mxGeometry x="40" y="18" width="1180" height="34" as="geometry" />\n'
        f'        </mxCell>',
    ]
    for name, (cx, rank, kind) in NODES.items():
        parts.append(box(name, cx, Y(rank), kind))
    for i, (src, tgt, dashed, wps) in enumerate(EDGES):
        parts.append(edge(i, src, tgt, dashed, wps))
    # легенда
    parts.append(
        '        <mxCell id="legend" value="── сплошная: фиксированный переход (add_edge)        '
        '┄┄ пунктир: условный переход (add_conditional_edges)        ⟲ self: ретрай шага" '
        'style="text;html=1;fontSize=13;align=left;verticalAlign=middle;fillColor=#ffffff;strokeColor=#cccccc;" '
        'vertex="1" parent="1">\n'
        '          <mxGeometry x="40" y="2300" width="1180" height="28" as="geometry" />\n'
        '        </mxCell>')
    parts += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>', '']
    open(dst, "w", encoding="utf-8").write("\n".join(parts))
    print(f"→ {dst} ({len(NODES)} узлов, {len(EDGES)} рёбер)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
