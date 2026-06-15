"""Рендер drawio (mxGraphModel) → статичный SVG для встраивания в README.

    python scripts/drawio_to_svg.py docs/architecture.drawio docs/architecture.svg

Узлы: rounded-rect / ellipse / text, многострочный value (&#10;), заливка/обводка,
пунктир, выравнивание, авто-подбор шрифта. Рёбра: ОРТОГОНАЛЬНЫЕ со СКРУГЛЁННЫМИ углами
(точки крепления exitX/exitY/entryX/entryY как доля 0..1 по границе + явные путевые точки
<mxGeometry><Array as="points">). Стрелки наследуют цвет ребра. Без точек крепления —
обрезка по лучу из центра. Этого достаточно для наших схем (граф агента и пр.).
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from html import escape, unescape
from math import hypot


def parse_style(s: str) -> dict:
    d = {}
    for part in (s or "").split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
        else:
            d[part] = "1"
    return d


def text_lines(val: str) -> list[str]:
    val = val.replace("&#10;", "\n").replace("<br>", "\n")
    val = re.sub(r"<[^>]+>", "", val)
    return [unescape(x) for x in val.split("\n")]


def rounded_path(pts: list[tuple[float, float]], r: float = 12.0) -> str:
    """SVG-path через точки со скруглёнными углами (квадратичные дуги в вершинах)."""
    if len(pts) < 2:
        return ""
    if len(pts) == 2:
        return f"M {pts[0][0]:.1f},{pts[0][1]:.1f} L {pts[1][0]:.1f},{pts[1][1]:.1f}"
    d = [f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        v1x, v1y = p1[0] - p0[0], p1[1] - p0[1]
        v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]
        l1, l2 = hypot(v1x, v1y) or 1, hypot(v2x, v2y) or 1
        rr = min(r, l1 / 2, l2 / 2)
        ax, ay = p1[0] - v1x / l1 * rr, p1[1] - v1y / l1 * rr
        bx, by = p1[0] + v2x / l2 * rr, p1[1] + v2y / l2 * rr
        d.append(f"L {ax:.1f},{ay:.1f}")
        d.append(f"Q {p1[0]:.1f},{p1[1]:.1f} {bx:.1f},{by:.1f}")
    d.append(f"L {pts[-1][0]:.1f},{pts[-1][1]:.1f}")
    return " ".join(d)


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    root = ET.parse(src).getroot()
    cells = list(root.iter("mxCell"))

    verts, edges, geo = {}, [], {}
    for c in cells:
        g = c.find("mxGeometry")
        st = parse_style(c.get("style", ""))
        if c.get("vertex") == "1" and g is not None and g.get("x") is not None:
            x, y = float(g.get("x")), float(g.get("y"))
            w, h = float(g.get("width", 120)), float(g.get("height", 40))
            geo[c.get("id")] = (x, y, w, h)
            verts[c.get("id")] = (c.get("value", ""), st, (x, y, w, h))
        elif c.get("edge") == "1":
            edges.append((c.get("source"), c.get("target"), st, c.get("value", ""), c))

    maxx = max((x + w for x, y, w, h in geo.values()), default=800) + 40
    maxy = max((y + h for x, y, w, h in geo.values()), default=600) + 40

    def center(i):
        x, y, w, h = geo[i]
        return (x + w / 2, y + h / 2)

    def conn(i, fx, fy):
        x, y, w, h = geo[i]
        return (x + fx * w, y + fy * h)

    def clip(cx, cy, i):
        x, y, w, h = geo[i]
        mx, my = x + w / 2, y + h / 2
        dx, dy = cx - mx, cy - my
        if dx == 0 and dy == 0:
            return mx, my
        sx = (w / 2) / abs(dx) if dx else 1e9
        sy = (h / 2) / abs(dy) if dy else 1e9
        s = min(sx, sy)
        return mx + dx * s, my + dy * s

    def waypoints(cell):
        pts = []
        for arr in cell.findall("mxGeometry/Array"):
            if arr.get("as") == "points":
                for p in arr.findall("mxPoint"):
                    pts.append((float(p.get("x", 0)), float(p.get("y", 0))))
        return pts

    body, edge_colors = [], set()

    # рёбра — первыми (под узлами); скруглённые ортогональные пути
    for srcid, tgtid, st, val, cell in edges:
        if srcid not in geo or tgtid not in geo:
            continue
        wps = waypoints(cell)
        self_loop = srcid == tgtid
        if "exitX" in st:
            sp = conn(srcid, float(st["exitX"]), float(st.get("exitY", 0.5)))
        else:
            sp = clip(*(wps[0] if wps else center(tgtid)), srcid)
        if "entryX" in st:
            ep = conn(tgtid, float(st["entryX"]), float(st.get("entryY", 0.5)))
        else:
            aim = wps[-1] if wps else (sp if self_loop else center(srcid))
            ep = clip(*aim, tgtid)
        pts = [sp] + wps + [ep]
        stroke = st.get("strokeColor", "#555555")
        edge_colors.add(stroke)
        dash = ' stroke-dasharray="6,5"' if st.get("dashed") == "1" else ""
        arrow = "" if st.get("endArrow") == "none" else f' marker-end="url(#a_{stroke.lstrip("#")})"'
        body.append(f'<path d="{rounded_path(pts)}" fill="none" stroke="{stroke}" '
                    f'stroke-width="1.7"{dash}{arrow}/>')
        if val:
            seg = max(range(len(pts) - 1),
                      key=lambda k: abs(pts[k + 1][0] - pts[k][0]) + abs(pts[k + 1][1] - pts[k][1]))
            mx, my = (pts[seg][0] + pts[seg + 1][0]) / 2, (pts[seg][1] + pts[seg + 1][1]) / 2
            body.append(f'<rect x="{mx - len(val) * 3.1 - 3:.0f}" y="{my - 13:.0f}" '
                        f'width="{len(val) * 6.2 + 6:.0f}" height="15" rx="3" fill="#ffffff" opacity="0.9"/>')
            body.append(f'<text x="{mx:.0f}" y="{my - 2:.0f}" font-size="10" fill="{stroke}" '
                        f'text-anchor="middle">{escape(val)}</text>')

    # узлы
    for vid, (val, st, (x, y, w, h)) in verts.items():
        fill = st.get("fillColor", "none")
        stroke = st.get("strokeColor", "#888888")
        dash = ' stroke-dasharray="6,4"' if st.get("dashed") == "1" else ""
        is_text = st.get("text") == "1"
        if is_text:
            pass
        elif "ellipse" in st:
            body.append(f'<ellipse cx="{x+w/2:.0f}" cy="{y+h/2:.0f}" rx="{w/2:.0f}" ry="{h/2:.0f}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
        else:
            body.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="9" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')

        bold = ' font-weight="bold"' if st.get("fontStyle") in ("1", "3") else ""
        valign = st.get("verticalAlign", "middle")
        align = st.get("align", "center")
        pad = float(st.get("spacingLeft", 8)) if align == "left" else 10
        maxw = max(4, w - 2 * pad)
        raws = text_lines(val)

        def wrap_at(fs):
            limit = max(6, int(maxw / (0.56 * fs)))
            ls = []
            for raw in raws:
                if len(raw) <= limit:
                    ls.append(raw); continue
                cur = ""
                for word in raw.split(" "):
                    if cur and len(cur) + 1 + len(word) > limit:
                        ls.append(cur); cur = word
                    else:
                        cur = (cur + " " + word).strip()
                if cur:
                    ls.append(cur)
            return ls

        start_fs = int(st.get("fontSize", 12))
        fs = start_fs
        while fs > 7:
            lines = wrap_at(fs)
            if len(lines) * (fs + 3) <= h - 4 or is_text:
                break
            fs -= 1
        else:
            lines = wrap_at(fs)
        lh = fs + 3
        total = len(lines) * lh
        if valign == "top":
            ty = y + fs + 4
        elif valign == "bottom":
            ty = y + h - total + fs
        else:
            ty = y + h / 2 - total / 2 + fs
        if align == "left":
            tx = x + float(st.get("spacingLeft", 8)); anchor = "start"
        elif align == "right":
            tx = x + w - 6; anchor = "end"
        else:
            tx = x + w / 2; anchor = "middle"
        for ln in lines:
            body.append(f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="{fs}" fill="#1a1a1a" '
                        f'text-anchor="{anchor}"{bold}>{escape(ln)}</text>')
            ty += lh

    # сборка: маркеры-стрелки по цвету ребра
    markers = "".join(
        f'<marker id="a_{c.lstrip("#")}" markerWidth="9" markerHeight="9" refX="7.5" refY="3" '
        f'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 z" fill="{c}"/></marker>'
        for c in sorted(edge_colors)
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{maxx:.0f}" height="{maxy:.0f}" '
        f'viewBox="0 0 {maxx:.0f} {maxy:.0f}" font-family="Helvetica,Arial,sans-serif">',
        f'<defs>{markers}</defs>',
        f'<rect width="{maxx:.0f}" height="{maxy:.0f}" fill="#ffffff"/>',
        *body,
        "</svg>",
    ]
    open(dst, "w", encoding="utf-8").write("\n".join(out))
    print(f"{src} → {dst} ({len(verts)} узлов, {len(edges)} рёбер, {maxx:.0f}×{maxy:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
