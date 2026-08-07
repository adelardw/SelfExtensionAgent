"""Бенч skill-конвейера (Hermes-порт), БЕЗ LLM. Три секции:
  1) СЕЛЕКТОР — hit@1/hit@3 ранкера навыков на фиксированных парах «запрос → ожидаемый навык»
     по РЕАЛЬНОМУ реестру (BM25 всегда; гибрид BM25+эмбеддинги — если есть ключ, ~15 embed-вызовов);
  2) ПЕСОЧНИЦА — накладные расходы изоляции: load-check навыка и python_exec в подпроцессе;
  3) КУРАТОР — сценарий жизненного цикла во временном каталоге (loser вычищен, winner/protected целы).
Запуск: uv run python bench_skills.py [--no-emb]
"""
import json
import os
import sys
import tempfile
import time

# Личные сторы не трогаем; реестр навыков читаем настоящий (секция 1), пишем только во temp (секция 3).
SELECTOR_CASES = [
    ("какая сейчас погода в Санкт-Петербурге", "weather_check"),
    ("распакуй скачанный zip архив", "zip_extractor"),
    ("молекулярная масса кофеина в pubchem", "pubchem_query"),
    ("сделай презентацию pptx на 5 слайдов", "generate_pptx"),
    ("найди публикации учёного по orcid", "orcid_reader"),
    ("разложи файлы в папке Загрузки по типам", "file_organizer"),
    ("установи python-пакет в проект", "uv_package_manager"),
    ("вытащи основной текст со страницы по ссылке", "link_parser"),
    ("найди в интернете свежие новости", "web_search"),
    ("цитаты твиттера в статьях википедии", "wikipedia_twitter_citations"),
]


def bench_selector(use_emb: bool) -> None:
    import src.tools.skill_creation as sc

    registry = sc._merged_registry()
    names, docs = [], []
    for n, meta in registry.items():
        md = sc._skill_base(n) / f"{n}.md"
        try:
            doc = md.read_text(encoding="utf-8")
        except OSError:
            doc = str(meta.get("description", ""))
        names.append(n)
        docs.append(f"{n} {doc}")

    def run(ranker_name: str, ranker) -> None:
        hit1 = hit3 = 0
        t0 = time.time()
        for q, want in SELECTOR_CASES:
            if want not in names:
                continue
            got = [names[i] for i in ranker(docs, q, 3)]
            hit1 += int(bool(got) and got[0] == want)
            hit3 += int(want in got)
        n = sum(1 for _, w in SELECTOR_CASES if w in names)
        dt = (time.time() - t0) / max(1, n)
        print(f"  {ranker_name:<8} hit@1 {hit1}/{n}  hit@3 {hit3}/{n}  {dt*1000:.0f}ms/запрос")

    print(f"[1] СЕЛЕКТОР — {len(names)} навыков в реестре")
    from src.search.retrieval import bm25_rank
    run("BM25", lambda d, q, k: bm25_rank(d, q, k))
    emb = sc._skill_embedder()
    if use_emb and getattr(emb, "enabled", False):
        run("Hybrid", lambda d, q, k: sc.rank_skill_docs(d, q, k, names=names))
    else:
        print("  Hybrid   пропущен (эмбеддер выключен/нет ключа или --no-emb)")


def bench_sandbox() -> None:
    from src.utils import _skill_loadable, run_python_sandboxed

    print("[2] ПЕСОЧНИЦА — накладные расходы изоляции")
    t0 = time.time()
    ok, msg = _skill_loadable("weather_check")
    print(f"  load-check навыка (subprocess): ok={ok} за {time.time()-t0:.2f}s ({msg[:60]})")
    t0 = time.time()
    ok, out = run_python_sandboxed("print(sum(range(10**6)))", timeout=20)
    print(f"  python_exec (subprocess+rlimits): ok={ok} за {time.time()-t0:.2f}s (out={out[:30]})")
    t0 = time.time()
    ok, out = run_python_sandboxed("while True: pass", timeout=3)
    print(f"  runaway-код: убит={not ok} за {time.time()-t0:.1f}s ({out[:60]})")


def bench_curator() -> None:
    import importlib

    import src.tools.skill_creation as sc

    print("[3] КУРАТОР — сценарий жизненного цикла (temp-каталог)")
    os.environ.pop("AGENT_EVAL_MODE", None)
    tmp = tempfile.mkdtemp(prefix="bench_skl_")
    old_dir, old_reg = sc.SKILLS_DIR, sc.REGISTRY_FILE
    try:
        sc.SKILLS_DIR = sc.Path(tmp)
        sc.REGISTRY_FILE = sc.SKILLS_DIR / "registry.json"
        sc.create_skill.invoke({"name": "loser", "description": "x", "scope": "global"})
        sc.create_skill.invoke({"name": "winner", "description": "x", "scope": "global"})
        sc.mark_temporary("winner")
        for _ in range(6):
            sc.record_skill_usage(["loser"], win=False)
            sc.record_skill_usage(["winner"], win=True)
        rep = sc.sync_registry()
        reg = sc._load_registry()
        ok = ("loser" in rep["curated_out"] and "winner" in reg
              and not reg["winner"].get("temporary"))
        print(f"  loser вычищен, winner принят делом: {'PASS' if ok else 'FAIL'} "
              f"(curated_out={rep['curated_out']})")
    finally:
        sc.SKILLS_DIR, sc.REGISTRY_FILE = old_dir, old_reg
        importlib.invalidate_caches()


if __name__ == "__main__":
    use_emb = "--no-emb" not in sys.argv
    if use_emb:
        from dotenv import load_dotenv

        load_dotenv()
    bench_selector(use_emb)
    bench_sandbox()
    bench_curator()
