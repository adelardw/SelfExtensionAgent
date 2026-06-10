"""
Живой прогресс прогона графа: человекочитаемые статусы по нодам.

Используется фронтендами (REPL/бот) поверх graph.astream(stream_mode=["updates","values"]):
на каждый update-шаг отдаёт короткий лейбл «что агент сейчас делает», чтобы при
долгих задачах (deliberate/heavy) было видно, что процесс идёт, и какой шаг плана
исполняется.
"""
from __future__ import annotations

NODE_LABELS = {
    "recall": "🔎 вспоминаю контекст",
    "goal": "🎯 определяю цель",
    "reflexion": "🤔 выбираю способ мышления",
    "fast_answer": "⚡ быстрый ответ",
    "reason": "🧠 размышляю",
    "router": "🧭 выбираю стратегию",
    "create_skills": "🧬 создаю навык",
    "sgr_create": "🧪 проверяю созданный навык",
    "skill_selector": "🧰 подбираю навыки",
    "capability_research": "🔬 ищу недостающие способности",
    "decompose": "🗂 раскладываю на подзадачи",
    "skill_injection": "💉 подключаю навыки",
    "synthesize": "🧩 собираю ответ",
    "review": "🏗 сквозной ревью решения",
    "validation": "✅ финальная валидация",
    "reflect": "📝 запоминаю",
}


class ProgressView:
    """Накапливает контекст прогона (план/шаги/режим) и строит лейбл текущего статуса."""

    def __init__(self) -> None:
        self.mode = ""
        self.n_steps = 0
        self.step_goals: list[str] = []

    def on_update(self, node: str, delta: dict) -> str | None:
        delta = delta or {}
        if node == "reflexion" and delta.get("mode"):
            self.mode = delta["mode"]
            return f"🤔 режим: {self.mode}"
        if node == "decompose":
            subs = delta.get("subtasks") or []
            self.n_steps = len(subs)
            self.step_goals = [s.get("goal", "") for s in subs]
            return f"🗂 план готов: {self.n_steps} шаг(ов)"
        if node == "step_executor":
            cur = delta.get("current_step")
            if cur is None:  # ретрай текущего шага
                return "🛠 шаг не прошёл проверку — ретрай"
            done = min(cur, self.n_steps or cur)
            goal = self.step_goals[cur - 1][:60] if 0 < cur <= len(self.step_goals) else ""
            return f"🛠 шаг {done}/{self.n_steps or '?'} готов" + (f": {goal}" if goal else "")
        if node == "review":
            subs = delta.get("subtasks")
            if subs and len(subs) > self.n_steps:
                added = len(subs) - self.n_steps
                self.n_steps = len(subs)
                self.step_goals = [s.get("goal", "") for s in subs]
                return f"🏗 ревью нашёл проблемы → +{added} шаг(а) доработки"
            return "🏗 ревью пройден"
        return NODE_LABELS.get(node)


async def stream_with_progress(graph, inputs: dict, config: dict, on_label) -> dict:
    """
    Прогоняет граф со стримингом: на каждый узел зовёт on_label(str) (может быть
    async), возвращает финальное состояние. on_label может игнорировать повторы.
    """
    import inspect

    pv = ProgressView()
    result: dict = {}
    async for stream_mode, payload in graph.astream(inputs, config=config, stream_mode=["updates", "values"]):
        if stream_mode == "values":
            result = payload
            continue
        for node, delta in (payload or {}).items():
            label = pv.on_update(node, delta if isinstance(delta, dict) else {})
            if label:
                r = on_label(label)
                if inspect.isawaitable(r):
                    await r
    return result
