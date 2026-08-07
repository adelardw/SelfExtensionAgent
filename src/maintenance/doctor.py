"""`sea doctor` — диагностика ВСЕХ известных болей одним прогоном.

Родилась из мульти-агентной валидации (2026-07-29): половина «дефектов агента» оказалась
тихими отказами ОКРУЖЕНИЯ (SearXNG «жив, но апстримы в капче» → 0 результатов при HTTP 200;
сломанный mcp-server-fetch; лежащий поиск), которые агент маскировал graceful-фолбэками.
Doctor делает их ВИДИМЫМИ до того, как они испортят прогоны.

Каждая проверка возвращает {name, status: ok|warn|fail|skip, detail, hint} — hint говорит,
ЧТО делать. Сетевые проверки — с короткими таймаутами; `--offline` их пропускает;
`--json` — машиночитаемый вывод. Exit code 1, если есть fail.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_ICON = {OK: "✅", WARN: "⚠️ ", FAIL: "❌", SKIP: "⏭ "}


def _r(name: str, status: str, detail: str = "", hint: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


# ── СВЯЗЬ / LLM ────────────────────────────────────────────────────────────────

def check_api_key() -> dict:
    from src.llm.llm import api_key, api_key_source

    if api_key():
        return _r("API-ключ", OK, f"источник: {api_key_source()}")
    return _r("API-ключ", FAIL, "не найден ни в env/.env, ни в user-config",
              "задай: `sea key` (скрытый ввод) или OPEN_ROUTER_API_KEY в .env")


def check_llm_endpoint(offline: bool) -> dict:
    if offline:
        return _r("LLM endpoint", SKIP, "--offline")
    from src.llm.llm import api_key, validate_credentials

    if not api_key():
        return _r("LLM endpoint", SKIP, "нет ключа")
    try:
        ok, msg = validate_credentials()
        return _r("LLM endpoint", OK if ok else FAIL, str(msg)[:120],
                  "" if ok else "проверь ключ/базовый URL: `sea config`, `sea key`")
    except Exception as e:  # noqa: BLE001
        return _r("LLM endpoint", FAIL, f"{type(e).__name__}: {e}"[:120],
                  "проверь сеть и ключ (`sea config`)")


def check_embeddings(offline: bool) -> dict:
    try:
        from omegaconf import OmegaConf

        mc = OmegaConf.load("config.yml").get("memory", {})
        if not mc.get("embeddings", False):
            return _r("Эмбеддинги", WARN, "выключены в config.yml (memory.embeddings=false)",
                      "recall/селектор навыков работают на token-overlap/BM25 — слабее")
        if offline:
            return _r("Эмбеддинги", SKIP, "--offline (включены в конфиге)")
        from src.memory.embedder import build_embedder

        emb = build_embedder(True, mc.get("embedding_model"))
        if not getattr(emb, "enabled", False):
            return _r("Эмбеддинги", FAIL, "включены в конфиге, но эмбеддер не поднялся (нет ключа?)",
                      "нужен OPEN_ROUTER_API_KEY (или OPENAI_API_KEY)")
        v = emb.embed("ping")
        return (_r("Эмбеддинги", OK, f"модель {getattr(emb, 'model', '?')}, dim={len(v)}")
                if v else _r("Эмбеддинги", FAIL, "embed() вернул пусто",
                             "проверь ключ/квоту OpenRouter"))
    except Exception as e:  # noqa: BLE001
        return _r("Эмбеддинги", FAIL, f"{type(e).__name__}: {e}"[:120])


# ── ПОИСК (боли валидации: «жив, но пуст» ≠ «работает») ────────────────────────

def classify_searxng(base: str, timeout: float = 6.0) -> tuple[str, str]:
    """(status, detail) по СОДЕРЖИМОМУ, не по коду ответа: not_set / down / alive_empty / ok."""
    if not base:
        return "not_set", "SEARXNG_URL не задан"
    url = f"{base.rstrip('/')}/search?q=test&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                   "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        return "down", f"{type(e).__name__}: {e}"[:100]
    n = len(data.get("results") or [])
    if n == 0:
        return "alive_empty", "HTTP 200, но 0 результатов на «test» — апстрим-движки в капче/бане"
    return "ok", f"{n} результатов на «test»"


def check_searxng(offline: bool) -> dict:
    if offline:
        return _r("SearXNG", SKIP, "--offline")
    status, detail = classify_searxng(os.getenv("SEARXNG_URL", ""))
    if status == "not_set":
        return _r("SearXNG", WARN, detail, "опционально: свой инстанс + SEARXNG_URL в .env "
                                           "(иначе поиск живёт на DDG-фолбэке)")
    if status == "down":
        return _r("SearXNG", FAIL, detail, "подними контейнер SearXNG (или убери SEARXNG_URL)")
    if status == "alive_empty":
        return _r("SearXNG", FAIL, detail,
                  "перезапусти контейнер/подожди снятия капчи; поиск сейчас едет на DDG-фолбэке")
    return _r("SearXNG", OK, detail)


def check_ddg_fallback(offline: bool) -> dict:
    if offline:
        return _r("DDG-фолбэк", SKIP, "--offline")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ws_doctor", "src/skills/web_search/web_search.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        res = m._search_fallback("погода москва", 3, "")
        if res:
            return _r("DDG-фолбэк", OK, f"{len(res)} результатов")
        return _r("DDG-фолбэк", FAIL, "0 результатов (бан/капча DDG?)",
                  "поиск без SearXNG И без DDG слеп — подожди или смени сеть/IP")
    except Exception as e:  # noqa: BLE001
        return _r("DDG-фолбэк", FAIL, f"{type(e).__name__}: {e}"[:100],
                  "поиск без SearXNG И без DDG слеп")


def check_browse(offline: bool) -> dict:
    if offline:
        return _r("Чтение страниц (browse)", SKIP, "--offline")
    try:
        import asyncio
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ws_doctor2", "src/skills/web_search/web_search.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out = asyncio.run(m.read_url.ainvoke({"url": "https://example.com", "max_chars": 500}))
        if isinstance(out, str) and len(out) > 80:
            return _r("Чтение страниц (browse)", OK, f"{len(out)} символов с example.com")
        return _r("Чтение страниц (browse)", WARN, f"короткий ответ: {str(out)[:80]}")
    except Exception as e:  # noqa: BLE001
        return _r("Чтение страниц (browse)", FAIL, f"{type(e).__name__}: {e}"[:100])


# ── ИНСТРУМЕНТЫ / ИЗОЛЯЦИЯ ─────────────────────────────────────────────────────

def check_mcp_server(name: str, spec: dict, timeout: float = 25.0) -> dict:
    """Поднимаем stdio-сервер и смотрим, не умер ли он с трейсбеком (ловит ImportError
    несовместимых версий — живой баг mcp-server-fetch/McpError)."""
    cmd = [spec["command"], *spec.get("args", [])]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        return _r(f"MCP {name}", FAIL, f"{spec['command']} не найден в PATH",
                  "установи uv (uvx) — https://docs.astral.sh/uv/")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:  # умер сам = ошибка старта
            err = (proc.stderr.read() or "")[-400:]
            frag = err.strip().splitlines()[-1] if err.strip() else f"rc={proc.returncode}"
            return _r(f"MCP {name}", FAIL, frag[:140],
                      "несовместимость версий? для fetch держим пин `--with mcp<2`")
        time.sleep(0.3)
        if time.time() - t0 > 2.0:  # 2с жив и молчит → сервер поднялся, ждёт stdio
            break
    proc.terminate()
    return _r(f"MCP {name}", OK, "поднимается (stdio ждёт клиента)")


def check_python_sandbox() -> dict:
    from src.utils import _syscall_sandbox_prefix, run_python_sandboxed

    ok, out = run_python_sandboxed("print(1+1)", timeout=20)
    prefix = _syscall_sandbox_prefix()
    iso = prefix[0] if prefix else "только rlimits (нет bwrap/sandbox-exec)"
    if ok and out.strip() == "2":
        return _r("Песочница python", OK, f"исполнение ок · syscall-изоляция: {iso}")
    return _r("Песочница python", FAIL, f"{out[:100]}",
              "python_exec/вычисления не работают — проверь системный python3")


def check_skill_loadcheck() -> dict:
    from src.utils import _skill_loadable

    ok, msg = _skill_loadable("web_search")
    return (_r("Load-check навыка (subprocess)", OK, msg[:80]) if ok
            else _r("Load-check навыка (subprocess)", FAIL, msg[:120]))


# ── ДАННЫЕ / ТЕЛЕМЕТРИЯ ────────────────────────────────────────────────────────

CKPT_BLOAT_MB = 150  # чекпоинты растут с каждым тредом; за порогом — пора чистить


def check_dbs() -> dict:
    parts, missing = [], []
    hint, st = "", OK
    for f in ("data/memory.db", "data/checkpoints.db", "data/traces.db"):
        p = Path(f)
        if p.exists():
            mb = p.stat().st_size / 1024 / 1024
            parts.append(f"{p.name} {mb:.0f}МБ")
            if p.name == "checkpoints.db" and mb > CKPT_BLOAT_MB:
                st = WARN
                hint = (f"checkpoints.db {mb:.0f}МБ (> {CKPT_BLOAT_MB}МБ) — распух от истории "
                        "шагов; `sea doctor --fix` оставит последний чекпоинт каждого треда "
                        "(с бэкапом .bak)")
        else:
            missing.append(p.name)
    if missing:
        st = WARN if st == OK else st
    detail = " · ".join(parts) + (f" · нет: {', '.join(missing)} (создадутся)" if missing else "")
    return _r("Базы данных", st, detail, hint)


def check_trace_pains(hours: float = 24.0) -> list[dict]:
    """Боли из телеметрии: ретрай-штормы и ноды с ошибками за последние N часов."""
    out = []
    try:
        from src.tracing.tracer import TraceStore

        ts = TraceStore()
        # run_id='unknown' — АГРЕГАТ прогонов без request_scope (CLI one-shot/драйвер): сумма
        # их шагов выглядит «штормом», не являясь им. Судим только идентифицированные прогоны.
        storms = [s for s in ts.retry_storms(hours) if dict(s).get("run_id") != "unknown"]
        if storms:
            frag = "; ".join(f"{dict(s)}"[:70] for s in storms[:3])
            out.append(_r("Ретрай-штормы (24ч)", WARN, frag,
                          "смотри трейс прогона: узел зациклился на ретраях"))
        else:
            out.append(_r("Ретрай-штормы (24ч)", OK, "не обнаружены (агрегат unknown не судим)"))
        rows = [dict(r) for r in ts.node_stats(hours)]
        bad = [r for r in rows if r.get("errors", r.get("err", 0))]
        if bad:
            frag = "; ".join(
                f"{r.get('node')}: {r.get('errors', r.get('err'))} err" for r in bad[:4])
            out.append(_r("Ошибки нод (24ч)", WARN, frag, "AGENT_DEBUG=1 + повтори сценарий"))
        else:
            out.append(_r("Ошибки нод (24ч)", OK, f"{len(rows)} нод без ошибок"))
        ts.close()
    except Exception as e:  # noqa: BLE001
        out.append(_r("Телеметрия traces.db", WARN, f"{type(e).__name__}: {e}"[:80]))
    return out


def check_skills() -> list[dict]:
    out = []
    try:
        from omegaconf import OmegaConf

        import src.tools.skill_creation as sc

        reg = sc._merged_registry()
        temp = [n for n, m in reg.items() if m.get("temporary")]
        broken = [n for n in reg
                  if reg[n].get("has_tools") and not (sc._skill_base(n) / f"{n}.py").exists()]
        skcfg = OmegaConf.load("config.yml").get("skills", {})
        min_uses = int(skcfg.get("curator_min_uses", 5))
        floor = float(skcfg.get("curator_win_floor", 0.2))
        losers = [n for n, m in reg.items()
                  if not m.get("protected") and not m.get("imported")
                  and int(m.get("uses", 0)) >= min_uses
                  and int(m.get("wins", 0)) / max(1, int(m.get("uses", 0))) < floor]
        detail = f"{len(reg)} в реестре · temp: {len(temp)} · кандидаты куратора: {len(losers)}"
        st = OK if not broken else FAIL
        if broken:
            detail += f" · БИТЫЕ (нет .py): {', '.join(broken[:4])}"
        out.append(_r("Реестр навыков", st, detail,
                      "битые: `sync_registry()` вычистит при старте" if broken else ""))
        # ЗАГРУЖАЕМОСТЬ: навык с has_tools, из которого не грузится НИ ОДНОГО тула, МЁРТВ
        # молча (вскрыто судьями: browser_control/launcher резались AST-гейтом из-за отсутствия
        # в protected — физ-браузер был мёртв во всех прогонах, наружу — только спам в логах).
        dead = []
        for n, m in reg.items():
            if not m.get("has_tools"):
                continue
            try:
                if not sc.get_all_loaded_skill_tools([n]):
                    dead.append(n)
            except Exception:  # noqa: BLE001
                dead.append(n)
        out.append(_r("Загружаемость навыков", FAIL if dead else OK,
                      ("МЕРТВЫ (0 тулов): " + ", ".join(dead[:6])) if dead
                      else f"{sum(1 for m in reg.values() if m.get('has_tools'))} навыков грузятся",
                      "authored-навык с subprocess/__import__ добавь в skills.protected; "
                      "сгенерированный — чини код" if dead else ""))
        # здоровье кода навыков (деградировавшие по N сбоев подряд)
        hp = Path("data/skill_health.json")
        if hp.exists():
            h = json.loads(hp.read_text("utf-8") or "{}")
            sick = [n for n, v in h.items()
                    if int(v.get("consecutive_failures", v.get("fails_row", 0))) >= 3]
            out.append(_r("Здоровье навыков", WARN if sick else OK,
                          ("деградируют: " + ", ".join(sick[:5])) if sick
                          else f"{len(h)} под наблюдением, деградаций нет",
                          "смотри data/skill_health.json (класс ошибки/аргументы)" if sick else ""))
    except Exception as e:  # noqa: BLE001
        out.append(_r("Реестр навыков", WARN, f"{type(e).__name__}: {e}"[:80]))
    return out


def check_config_sanity() -> dict:
    try:
        from omegaconf import OmegaConf

        import src.tools.skill_creation as sc

        cfg = OmegaConf.load("config.yml")
        missing = [n for n in (cfg.get("skills", {}).get("protected", []) or [])
                   if not (sc.SKILLS_DIR / n).is_dir()]
        risky = os.getenv("AGENT_ALLOW_RISKY_SKILLS") == "1"
        detail = []
        if missing:
            detail.append(f"protected без папки: {', '.join(missing[:5])}")
        if risky:
            detail.append("AGENT_ALLOW_RISKY_SKILLS=1 — AST-гейт ВЫКЛЮЧЕН")
        if not detail:
            return _r("Конфиг", OK, "protected-навыки на месте, гейты включены")
        return _r("Конфиг", WARN if not missing else FAIL, " · ".join(detail),
                  "убери лишние protected из config.yml / сними AGENT_ALLOW_RISKY_SKILLS")
    except Exception as e:  # noqa: BLE001
        return _r("Конфиг", WARN, f"{type(e).__name__}: {e}"[:80])


# ── сборка ─────────────────────────────────────────────────────────────────────

def run_checks(offline: bool = False, deep_mcp: bool = True) -> list[dict]:
    from dotenv import load_dotenv

    load_dotenv()
    results: list[dict] = []
    results.append(check_api_key())
    results.append(check_llm_endpoint(offline))
    results.append(check_embeddings(offline))
    results.append(check_searxng(offline))
    results.append(check_ddg_fallback(offline))
    results.append(check_browse(offline))
    if deep_mcp and not offline:
        from src.data.mcp_client import CATALOG

        for name, entry in CATALOG.items():
            results.append(check_mcp_server(name, entry["spec"]))
    results.append(check_python_sandbox())
    results.append(check_skill_loadcheck())
    results.append(check_dbs())
    results.extend(check_trace_pains())
    results.extend(check_skills())
    results.append(check_config_sanity())
    return results


def render(results: list[dict]) -> str:
    lines = ["sea doctor — диагностика болей", "─" * 64]
    for r in results:
        line = f"{_ICON[r['status']]} {r['name']:<32} {r['detail']}"
        lines.append(line)
        if r["hint"] and r["status"] in (WARN, FAIL):
            lines.append(f"     ↳ {r['hint']}")
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_warn = sum(1 for r in results if r["status"] == WARN)
    lines.append("─" * 64)
    lines.append(f"итог: {len(results)} проверок · ❌ {n_fail} · ⚠️ {n_warn}")
    return "\n".join(lines)


ARTIFACT_TTL_DAYS = float(os.getenv("AGENT_ARTIFACT_TTL_DAYS") or 14)


def prune_artifacts(ttl_days: float = ARTIFACT_TTL_DAYS) -> int:
    """Удалить каталоги произведённых файлов старше TTL. Артефакты копятся от КАЖДОГО
    прогона (включая внутренние ретраи — валидация видела «мусор» с частичными таблицами),
    а нужны они пользователю в момент выдачи. Свежие не трогаем."""
    import shutil
    import time as _t

    root = Path("artifacts")
    if not root.is_dir():
        return 0
    cutoff = _t.time() - ttl_days * 86400
    removed = 0
    for d in root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d)
                removed += 1
        except OSError:
            pass
    return removed


def apply_fixes() -> list[str]:
    """Безопасные авто-фиксы (`sea doctor --fix`): прунинг распухших чекпоинтов (с бэкапом
    .bak; теряется только ИСТОРИЯ шагов тредов, не их текущее состояние) + чистка старых
    артефактов (файлы старше TTL уже доставлены пользователю)."""
    done: list[str] = []
    p = Path("data/checkpoints.db")
    if p.exists() and p.stat().st_size / 1024 / 1024 > CKPT_BLOAT_MB:
        from src.maintenance.db_prune import prune_checkpoints

        r = prune_checkpoints(str(p))
        done.append(f"checkpoints.db: {r['before_mb']}МБ → {r['after_mb']}МБ "
                    f"(-{r['dropped_checkpoints']} чекпоинтов, бэкап {r['backup']})")
    n = prune_artifacts()
    if n:
        done.append(f"artifacts/: удалено {n} каталог(ов) старше {ARTIFACT_TTL_DAYS:.0f} дней")
    return done


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    offline = "--offline" in args
    as_json = "--json" in args
    if "--fix" in args:
        fixed = apply_fixes()
        for f in fixed:
            print(f"🔧 {f}")
        if not fixed:
            print("🔧 авто-фиксов не потребовалось")
    results = run_checks(offline=offline)
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(render(results))
    return 1 if any(r["status"] == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
