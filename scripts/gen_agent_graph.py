"""Схема РЕАЛЬНОГО графа агента (src/agent.py) → .drawio (→ SVG через drawio_to_svg.py).

Структура — РОВНО как выдаёт LangGraph (.get_graph().draw_*): те же ноды и те же рёбра
(сплошные = add_edge, пунктирные = add_conditional_edges), та же слоёная раскладка
(спайн-вход справа, пайплайн deliberate слева, обратные рёбра — по боковым полосам).
НО вместо голых имён в каждой ноде — ПОДРОБНОЕ описание (что/как идёт), как в ASCII из
ARCHITECTURE.md. Крупно, не жалея пространства.

    python scripts/gen_agent_graph.py docs/architecture.en.drawio en "<title>"
    python scripts/drawio_to_svg.py   docs/architecture.en.drawio docs/architecture.en.svg
"""
from __future__ import annotations

import sys
from html import escape

ROWH = 290
TOP = 440                                     # место сверху под entrypoints
C1, C2, C3, C4 = 480, 1020, 1560, 2080       # пайплайн / ветка / спайн-вход / fast·reason
NW, NH = 380, 140                            # бокс-ноды
def Y(r): return TOP + r * ROWH

# entrypoints (app-слой) — вызывают граф; (label, desc_en, desc_ru, центр_x)
EY = 210
ENTRY = [
    ("CLI · sea  (default surface)",
     "full-screen TUI (Textual): slash-cmds + ghost/Tab,\n"
     "↑/↓ history, Esc-close chat, mouse select + scroll.\n"
     "HITL confirm + clarify Q/A pickers (❯  [✓], Enter ticks → Done).\n"
     "live tokens (⟳ calling model) · browser bridge (/token) ·\n"
     "/init repo→SEA.md · /compact·/sync · one-shot: sea \"task\"",
     "полноэкранный TUI (Textual): слэш-команды + ghost/Tab,\n"
     "↑/↓ история, Esc-закрыть чат, мышь выделяет + скроллит.\n"
     "HITL-подтверждение + Q/A-пикеры (❯  [✓], Enter ставит ✓ → Done).\n"
     "живые токены (⟳ calling model) · мост браузера (/token) ·\n"
     "/init репо→SEA.md · /compact·/sync · one-shot: sea «задача»", 760),
    ("Telegram", "bot.py · chat + clarify/confirm", "bot.py · чат + clarify/confirm", 1280),
    ("Desktop / Web GUI", "desktop.py · React+Vite\nlive progress (astream)",
     "desktop.py · React+Vite\nживой прогресс (astream)", 1700),
    ("Chrome extension", "agent in your browser · side-panel chat\n→ browser_bridge → same graph",
     "агент в твоём браузере · чат из side-panel\n→ browser_bridge → тот же граф", 2120),
    ("FastAPI server", "server.py · /chat · /run · astream", "server.py · /chat · /run · astream", 2540),
]

# имя → (центр_x, rank, тип)   тип: "io" | "node"
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

# подробное описание «что и как идёт» (заголовок-имя движок делает жирным сам)
DESC = {
    "en": {
        "recall": "memory (episodes / facts / conclusions / goals)\n+ implicit feedback + external ctx + AutoRAG\n(user KB + session files, BM25 + sanitize).\nCONDITIONAL & FLEXIBLE (recall_gate): facts AND\nassociative memory BY RELEVANCE · GraphRAG-lite ·\nquery embedded ONCE → router reuses it.\n+ inject SEMANTICALLY-closest session_findings\n(cached digests of past HEAVY runs)",
        "reflexion": "Self-Reflexion CHOICE of the thinking mode from\ntask analysis · + bandit prior (Beta/Thompson over\nthe user's episodes, sees failures) + few-shots\n+ intent router L2 (embedding-kNN, any language).\nhigh ambiguity → clarify · heavy is NOT predicted —\nit is EARNED by runtime evidence.\nfindings already cover it → COMPRESS the mode\n(deliberate → reason/fast); re-escalate on evidence",
        "act": "System 1 with hands: ONE direct action with\n1–2 tools (BM25 skill pick, HITL kept).\nzero tool calls / ESCALATE → goal (deliberate path)",
        "goal": "goal-setting: aim + a 'worthwhile' goal +\nrubric, kept in context for all nodes",
        "fast_answer": "direct answer (System 1 / System 2), no tools;\nthe clarify branch also resolves here",
        "clarify_gate": "batch of precise clarifications (markers where the\nset is finite, open otherwise). answers → the run's\nclarification registry (dedup, reused by\ndecompose / step / synthesize)",
        "reason": "System 2: deep step-by-step reasoning,\nno tools → goes through final validation",
        "router": "route the deliberate path: build a new skill\n(create_skills) or pick existing ones (skill_selector)",
        "create_skills": "ReAct builds a new skill (code) for a\ncapability gap → into the skill library",
        "sgr_create": "SGR review + smoke test of the new skill →\nload it (L1 self-improvement) or back to router",
        "skill_selector": "BM25 retrieval of relevant skills for the query.\nAMORTIZATION: with a PATTERN runs with NO LLM call",
        "capability_research": "research capabilities; on a gap discover & connect\nan MCP server (registry + trusted catalog, HITL)",
        "decompose": "plan the task into steps → skill_injection.\nat sim ≥ 0.7 → NO LLM (plan straight from a pattern)",
        "skill_injection": "inject the selected skills / tools into\nthe step executor's toolset",
        "step_executor": "execute + validate PER ITEM (⟲ retry / next step).\nthe validator sees the ACTUALLY-called tools\n(text ≠ action) · within-step context masking ·\nask_user catch-up at a fork",
        "synthesize": "assemble the final solution from the steps",
        "review": "HEAVY only: end-to-end review of the assembled\nsolution by the DEEP model · problems → fix\nsub-steps (→ step_executor) · clean → validation",
        "validation": "grounding + rubric check of the answer\n(deterministic anti-hallucination floor)",
        "reflect": "write the episode (+ interaction journal), harvest\nsignal (HITL / clarify → profile facts), compile the\nPATTERN (+win/lose) → collective pool, detect a\nhabit, extract facts (+edges), summary, prune,\ndegradation tracking, auto self-learning.\nHEAVY/multi-step run → COMPRESS into a findings\ndigest → session_findings (state, by thread)",
    },
    "ru": {
        "recall": "память (эпизоды / факты / выводы / цели)\n+ implicit feedback + external ctx + AutoRAG\n(БЗ юзера + файлы сессии, BM25 + sanitize).\nУСЛОВНЫЙ и ГИБКИЙ (recall_gate): И факты, И\nассоциативная память ПО РЕЛЕВАНТНОСТИ · GraphRAG-lite\nзапрос эмбеддится ОДИН раз → роутер переиспользует.\n+ впрыск СЕМАНТИЧЕСКИ ближайших session_findings\n(кэш выжимок прошлых ТЯЖЁЛЫХ прогонов)",
        "reflexion": "Self-Reflexion ВЫБОР режима мышления по анализу\nзадачи · + априор бандита (Beta/Thompson по эпизодам\nюзера, видит неудачи) + few-shots\n+ intent-роутер L2 (embedding-kNN, любой язык).\nвысокая неоднозначность → clarify · heavy НЕ\nпредсказывается — ЗАРАБАТЫВАЕТСЯ рантайм-evidence.\nнаходки уже покрывают → СЖАТЬ режим\n(deliberate → reason/fast); эскалация назад по evidence",
        "act": "System 1 с руками: ОДНО прямое действие\n1–2 инструментами (BM25-подбор навыка, HITL).\nноль вызовов / ESCALATE → goal (путь deliberate)",
        "goal": "целеполагание: цель + «стоящая» цель +\nrubric, держится в контексте всех нод",
        "fast_answer": "прямой ответ (System 1 / System 2), без тулов;\nветка clarify тоже отвечает здесь",
        "clarify_gate": "батч точных уточнений (маркеры где набор конечен,\nоткрытые где нет). ответы → реестр уточнений\nпрогона (дедуп, переиспользуют\ndecompose / step / synthesize)",
        "reason": "System 2: глубокое пошаговое рассуждение,\nбез тулов → проходит финальную validation",
        "router": "ветка пути deliberate: создать навык\n(create_skills) или взять готовые (skill_selector)",
        "create_skills": "ReAct строит новый навык (код) под пробел\nв способностях → в библиотеку навыков",
        "sgr_create": "SGR-ревью + smoke-тест нового навыка →\nзагрузка (L1 self-improvement) или назад в router",
        "skill_selector": "BM25-подбор релевантных навыков под запрос.\nАМОРТИЗАЦИЯ: при ПАТТЕРНЕ — БЕЗ LLM-вызова",
        "capability_research": "разведка способностей; на пробеле — найти и\nподключить MCP (реестр + доверенный каталог, HITL)",
        "decompose": "план задачи по шагам → skill_injection.\nпри sim ≥ 0.7 → БЕЗ LLM (план прямо из паттерна)",
        "skill_injection": "инъекция выбранных навыков / тулов\nв набор инструментов исполнителя шагов",
        "step_executor": "исполнение + валидация ПО ПУНКТАМ (⟲ ретрай / шаг).\nвалидатор видит РЕАЛЬНО вызванные тулы\n(текст ≠ действие) · маскинг контекста внутри шага ·\nдогон ask_user на развилке",
        "synthesize": "сборка финального решения из шагов",
        "review": "только HEAVY: сквозной ревью собранного решения\nDEEP-моделью · проблемы → fix-подшаги\n(→ step_executor) · чисто → validation",
        "validation": "заземление + проверка по rubric\n(детерминированный анти-галлюцинационный пол)",
        "reflect": "запись эпизода (+ журнал взаимодействий), harvest\nсигнала (HITL / clarify → факты профиля), компиляция\nПАТТЕРНА (+win/lose) → коллективный пул, детекция\nпривычки, извлечение фактов (+рёбра), саммари, prune,\nтрекинг деградации, авто self-learning.\nТЯЖЁЛЫЙ/мультишаговый прогон → СЖАТЬ в выжимку-\nfindings → session_findings (state, по треду)",
    },
}

# --- семантические цвета (стандартная палитра drawio) ---
COLORS = {
    "memory":    ("#E1D5E7", "#9673A6"),   # фиолетовый — память
    "routing":   ("#FFE6CC", "#D79B00"),   # оранжевый — выбор режима / маршрутизация / clarify
    "goal":      ("#B0E3E6", "#0E8088"),   # бирюзовый — цель
    "think":     ("#D5E8D4", "#82B366"),   # зелёный — мышление / исполнение
    "skill":     ("#DAE8FC", "#6C8EBF"),   # синий — навыки / способности
    "review":    ("#FFF2CC", "#D6B656"),   # жёлтый — ревью / самообучение
    "iface":     ("#F5F5F5", "#666666"),   # серый — entrypoints (app-слой)
    "start":     ("#D5E8D4", "#82B366"),
    "end":       ("#F8CECC", "#B85450"),
}
CAT = {
    "__start__": "start", "__end__": "end",
    "recall": "memory",
    "reflexion": "routing", "router": "routing", "clarify_gate": "routing",
    "goal": "goal",
    "act": "think", "fast_answer": "think", "reason": "think",
    "decompose": "think", "skill_injection": "think",
    "step_executor": "think", "synthesize": "think", "validation": "think",
    "create_skills": "skill", "sgr_create": "skill",
    "skill_selector": "skill", "capability_research": "skill",
    "review": "review", "reflect": "review",
}

# рёбра: (src, tgt, dashed?, waypoints[])  — ровно граф LangGraph
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
    ("reason", "validation", SOLID, [(2620, Y(4) + 70), (2620, Y(15))]),
    ("fast_answer", "reflect", SOLID, [(2540, Y(3) + 70), (2540, Y(16))]),
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
    ("sgr_create", "router", DASH, [(90, Y(7)), (90, Y(5))]),
    ("sgr_create", "create_skills", DASH, [(810, Y(7)), (810, Y(6))]),
    ("sgr_create", "skill_selector", DASH, []),
    ("step_executor", "step_executor", DASH, [(265, Y(12) - 36), (265, Y(12) + 36)]),
    ("synthesize", "review", DASH, []),
    ("synthesize", "validation", DASH, []),
    ("review", "step_executor", DASH, [(150, Y(14)), (150, Y(12))]),
    ("review", "validation", DASH, []),
    ("validation", "router", DASH, [(210, Y(15)), (210, Y(5))]),
    ("validation", "reflect", DASH, []),
]


def cell(cid, value, style, cx, cy, w, h):
    x, y = cx - w / 2, cy - h / 2
    val = escape(value).replace("\n", "&#10;")   # \n в XML-атрибуте схлопнулся бы в пробел → &#10;
    return (f'        <mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w}" height="{h}" as="geometry" />\n'
            f'        </mxCell>')


def box(name, cx, cy, kind, desc):
    fill, stroke = COLORS[CAT[name]]
    if kind == "io":
        style = (f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                 "fontSize=16;fontStyle=1;shadow=1;")
        return cell(name, name, style, cx, cy, 180, 64)
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             "fontSize=13;verticalAlign=middle;shadow=1;arcSize=10;")
    return cell(name, f"{name}\n{desc}" if desc else name, style, cx, cy, NW, NH)


def entry_box(i, label, desc, cx):
    fill, stroke = COLORS["iface"]
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             "fontSize=13;verticalAlign=middle;shadow=1;arcSize=12;align=left;spacingLeft=10;")
    # CLI-вход (i==0) — детальный, выше; остальные — компактные. cell() сам экранирует; первую
    # строку (label) рендер делает жирной — поэтому без <b> и без ручного escape здесь.
    h = 168 if i == 0 else 96
    return cell(f"ep{i}", f"{label}\n{desc}", style, cx, EY, 460, h)


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
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    title = sys.argv[3] if len(sys.argv) > 3 else "agent graph (LangGraph forward graph)"
    desc = DESC[lang]
    if lang == "en":
        legend = ("── solid: fixed transition (add_edge)        ┄┄ dashed: conditional (add_conditional_edges)        ⟲ self-loop: step retry / next step\n"
                  "colours — gray: entrypoints · purple: memory · orange: routing/clarify · teal: goal · green: thinking/execution · blue: skills/capabilities · yellow: review/self-learning")
    else:
        legend = ("── сплошная: фиксированный переход (add_edge)        ┄┄ пунктир: условный (add_conditional_edges)        ⟲ self: ретрай шага / следующий шаг\n"
                  "цвета — серый: entrypoints · фиолетовый: память · оранжевый: роутинг/clarify · бирюзовый: цель · зелёный: мышление/исполнение · синий: навыки · жёлтый: ревью/самообучение")
    parts = [
        '<mxfile host="app.diagrams.net" agent="self-extension-agent">',
        '  <diagram name="agent-graph" id="agent-graph">',
        '    <mxGraphModel dx="1600" dy="1100" grid="0" gridSize="10" guides="1" tooltips="1" '
        'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2200" pageHeight="3900" '
        'math="0" shadow="0">',
        '      <root>',
        '        <mxCell id="0" />',
        '        <mxCell id="1" parent="0" />',
        f'        <mxCell id="title" value="{escape(title)}" style="text;html=1;fontSize=22;fontStyle=1;align=left;verticalAlign=middle;" vertex="1" parent="1">\n'
        f'          <mxGeometry x="60" y="24" width="2100" height="44" as="geometry" />\n'
        f'        </mxCell>',
    ]
    # entrypoints: заголовок + боксы + рёбра в __start__
    ep_hdr = "ENTRYPOINTS (app layer) — invoke the graph (shared graph + memory)" if lang == "en" \
        else "ENTRYPOINTS (app-слой) — вызывают граф (общий граф + память)"
    parts.append(
        f'        <mxCell id="ephdr" value="{escape(ep_hdr)}" '
        'style="text;html=1;fontSize=15;fontStyle=1;align=left;verticalAlign=middle;" vertex="1" parent="1">\n'
        f'          <mxGeometry x="60" y="88" width="2100" height="26" as="geometry" />\n'
        '        </mxCell>')
    for i, (label, de, dr, cx) in enumerate(ENTRY):
        parts.append(entry_box(i, label, de if lang == "en" else dr, cx))
        parts.append(
            f'        <mxCell id="ep_e{i}" style="html=1;endArrow=block;rounded=1;strokeColor=#4a4a6a;" '
            f'edge="1" parent="1" source="ep{i}" target="__start__"><mxGeometry relative="1" as="geometry" /></mxCell>')
    for name, (cx, rank, kind) in NODES.items():
        parts.append(box(name, cx, Y(rank), kind, desc.get(name, "")))
    for i, (src, tgt, dashed, wps) in enumerate(EDGES):
        parts.append(edge(i, src, tgt, dashed, wps))
    parts.append(cell("legend", legend,
                      "text;html=1;fontSize=15;align=left;verticalAlign=top;fillColor=#ffffff;strokeColor=#cccccc;",
                      60 + 1050, Y(17) + 116, 2100, 64))
    parts += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>', '']
    open(dst, "w", encoding="utf-8").write("\n".join(parts))
    print(f"→ {dst} ({len(NODES)} узлов, {len(EDGES)} рёбер, lang={lang})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
