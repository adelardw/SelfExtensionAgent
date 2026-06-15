"""Рендер drawio (mxGraphModel) → статичный SVG для встраивания в README.

    python scripts/drawio_to_svg.py docs/architecture.drawio docs/architecture.svg

Поддерживает узлы (rounded-rect / ellipse / text), многострочный value (&#10;),
заливку/обводку, пунктир, выравнивание, жирный шрифт, и рёбра со стрелками.
Рёбра — ОРТОГОНАЛЬНЫЕ (polyline): точки крепления exitX/exitY/entryX/entryY (доля 0..1
по границе бокса) + явные путевые точки (<mxGeometry><Array as="points">). Без точек
крепления — обрезка по лучу из центра (как раньше). Этого достаточно для наших схем.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from html import escape


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
    val = re.sub(r"<[^>]+>", "", val)          # срезать html-теги
    from html import unescape
    return [unescape(x) for x in val.split("\n")]


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

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{maxx:.0f}" height="{maxy:.0f}" '
        f'viewBox="0 0 {maxx:.0f} {maxy:.0f}" font-family="Helvetica,Arial,sans-serif">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 z" fill="#555"/></marker></defs>',
        f'<rect width="{maxx:.0f}" height="{maxy:.0f}" fill="#ffffff"/>',
    ]

    def center(i):
        x, y, w, h = geo[i]
        return (x + w / 2, y + h / 2)

    def conn(i, fx, fy):  # точка крепления на границе бокса по доле (fx,fy) ∈ [0,1]
        x, y, w, h = geo[i]
        return (x + fx * w, y + fy * h)

    def clip(cx, cy, i):  # точка на границе прямоугольника i по лучу из его центра к (cx,cy)
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

    # рёбра — первыми (под узлами); ортогональные polyline через путевые точки
    for srcid, tgtid, st, val, cell in edges:
        if srcid not in geo or tgtid not in geo:
            continue
        wps = waypoints(cell)
        if "exitX" in st:
            sp = conn(srcid, float(st["exitX"]), float(st.get("exitY", 0.5)))
        else:
            sp = clip(*(wps[0] if wps else center(tgtid)), srcid)
        if "entryX" in st:
            ep = conn(tgtid, float(st["entryX"]), float(st.get("entryY", 0.5)))
        else:
            ep = clip(*(wps[-1] if wps else center(srcid)), tgtid)
        pts = [sp] + wps + [ep]
        stroke = st.get("strokeColor", "#555555")
        dash = ' stroke-dasharray="5,4"' if st.get("dashed") == "1" else ""
        arrow = "" if st.get("endArrow") == "none" else ' marker-end="url(#arrow)"'
        d = " ".join(f"{px:.0f},{py:.0f}" for px, py in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{stroke}" '
                   f'stroke-width="1.4"{dash}{arrow}/>')
        if val:
            # подпись — на середине самого длинного сегмента
            seg = max(range(len(pts) - 1),
                      key=lambda k: abs(pts[k+1][0]-pts[k][0]) + abs(pts[k+1][1]-pts[k][1]))
            mx, my = (pts[seg][0] + pts[seg+1][0]) / 2, (pts[seg][1] + pts[seg+1][1]) / 2
            out.append(f'<rect x="{mx-len(val)*2.7-3:.0f}" y="{my-12:.0f}" '
                       f'width="{len(val)*5.4+6:.0f}" height="13" fill="#ffffff" opacity="0.85"/>')
            out.append(f'<text x="{mx:.0f}" y="{my-2:.0f}" font-size="9" fill="{stroke}" '
                       f'text-anchor="middle">{escape(val)}</text>')

    # узлы
    for vid, (val, st, (x, y, w, h)) in verts.items():
        fill = st.get("fillColor", "none")
        stroke = st.get("strokeColor", "#888888")
        dash = ' stroke-dasharray="6,4"' if st.get("dashed") == "1" else ""
        is_text = st.get("text") == "1"
        if is_text:
            pass  # без рамки
        elif "ellipse" in st:
            out.append(f'<ellipse cx="{x+w/2:.0f}" cy="{y+h/2:.0f}" rx="{w/2:.0f}" ry="{h/2:.0f}" '
                       f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"{dash}/>')
        else:
            out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="8" '
                       f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"{dash}/>')

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

        # авто-подбор шрифта: уменьшаем, пока текст не влезет в бокс по высоте
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
            out.append(f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="{fs}" fill="#1a1a1a" '
                       f'text-anchor="{anchor}"{bold}>{escape(ln)}</text>')
            ty += lh

    out.append("</svg>")
    open(dst, "w", encoding="utf-8").write("\n".join(out))
    print(f"{src} → {dst} ({len(verts)} узлов, {len(edges)} рёбер, {maxx:.0f}×{maxy:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
