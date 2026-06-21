"""
Изоляция per-request состояния под КОНКУРЕНТНЫМ сервером (баг ревью: глобалы не изолированы по
run/user). Это тот тип теста, который однопоточные юниты не ловят — два interleaved-запроса не
должны затирать друг другу HITL-гранты, анти-тайпсквоттинг-домены и токен-бюджет.
"""
import asyncio

from src.runtime import run_context, runbudget, hitl
from src.browser import browser_bridge


def test_hitl_grants_isolated_per_user():
    async def worker(rid: str, uid: str, out: dict) -> None:
        with run_context.request_scope(rid, uid):
            hitl.grant(f"skill.tool_{uid}", persist=False)   # «да, всегда» ЭТОГО юзера
            await asyncio.sleep(0.01)                        # дать другому таску перемешаться
            out[uid] = {
                "own": hitl.is_granted(f"skill.tool_{uid}"),
                "other": hitl.is_granted("skill.tool_B" if uid == "A" else "skill.tool_A"),
            }

    async def main():
        hitl._user_grants.clear()
        out: dict = {}
        await asyncio.gather(worker("r1", "A", out), worker("r2", "B", out))
        return out

    out = asyncio.run(main())
    assert out["A"]["own"] and out["B"]["own"]               # свой грант виден
    assert not out["A"]["other"] and not out["B"]["other"]   # ЧУЖОЙ грант НЕ протёк (анти-эскалация)


def test_browser_domains_isolated_per_request():
    async def worker(rid: str, dom: str, out: dict) -> None:
        with run_context.request_scope(rid, "u"):
            browser_bridge.set_user_domains(f"открой {dom}")
            await asyncio.sleep(0.01)
            # typo домена → safe_url должен подставить домен ИМЕННО этого запроса
            typo = dom.replace(".com", ".cm")
            out[dom] = browser_bridge.safe_url(f"https://{typo}/x")

    async def main():
        out: dict = {}
        await asyncio.gather(worker("r1", "tanuki.com", out), worker("r2", "sakura.com", out))
        return out

    out = asyncio.run(main())
    assert "tanuki.com" in out["tanuki.com"]                 # каждый запрос видит СВОИ домены
    assert "sakura.com" in out["sakura.com"]


def test_browser_domains_visible_across_nodes_within_request():
    """−9pp-капкан: set_user_domains зовётся в recall (нода), safe_url — в ДРУГОЙ ноде. contextvar,
    выставленный в ноде, не виден сестре → было бы сломано. Общий dict по run_id (с границы) —
    видно между нодами одного запроса И изолировано между запросами."""
    async def main():
        with run_context.request_scope("r1", "u"):
            async def node_set():       # как recall
                browser_bridge.set_user_domains("открой tanuki.com")
            async def node_read():      # как act/step — ДРУГАЯ нода-таск
                return browser_bridge.safe_url("https://tanuki.cm/x")  # typo-домен
            await asyncio.create_task(node_set())
            return await asyncio.create_task(node_read())

    out = asyncio.run(main())
    assert "tanuki.com" in out          # домен из «recall» виден в «act» → анти-тайпсквоттинг жив


def test_grant_persist_gated_to_operator(monkeypatch):
    """#5: рантайм-«да, всегда» КЛИЕНТА сервера НЕ персистится глобально (утечка per-user→global
    после рестарта). Персист — привилегия оператора (uid='', REPL/desktop)."""
    from src.runtime import hitl
    import src.config.cli_config as cc
    persisted: list = []
    monkeypatch.setattr(cc, "set_cli", lambda k, v: persisted.append((k, v)))
    monkeypatch.setattr(cc, "get_cli", lambda k: [])
    hitl._user_grants.clear()

    hitl.grant("skill.op", persist=True)                 # оператор (нет scope, uid='')
    assert len(persisted) == 1                            # → персистит

    with run_context.request_scope("r1", "client42"):
        hitl.grant("skill.cl", persist=True)             # клиент сервера
    assert len(persisted) == 1                            # → НЕ персистит (сессионно)
    assert "skill.cl" in hitl._user_grants.get("client42", set())


def test_runbudget_isolated_under_interleave():
    async def worker(rid: str, tokens: int, out: dict) -> None:
        with run_context.request_scope(rid, "u"):
            runbudget.reset()
            runbudget.add(tokens)
            await asyncio.sleep(0.01)
            out[rid] = runbudget.used()                      # не затёрт другим прогоном

    async def main():
        out: dict = {}
        await asyncio.gather(worker("r1", 1000, out), worker("r2", 25, out))
        return out

    out = asyncio.run(main())
    assert out == {"r1": 1000, "r2": 25}
