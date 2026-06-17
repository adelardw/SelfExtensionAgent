"""Вычислительный слой: исполнение Python в песочнице. Офлайн, без LLM."""
from src.compute import make_compute_tool
from src.utils import run_python_sandboxed


def test_exact_computation():
    ok, out = run_python_sandboxed(
        "import statistics\n"
        "vals = [12.0, 8.5, 14.2, 9.1, 11.0]\n"
        "print(round(statistics.mean(vals), 2))")
    assert ok and out.strip() == "10.96"


def test_big_number_arithmetic():
    # ровно то, что LLM (и человек) делает ненадёжно — сверяем с самим Python
    ok, out = run_python_sandboxed("print(34689 * 1789 + 42)")
    assert ok and out.strip() == str(34689 * 1789 + 42)


def test_syntax_error_handled():
    ok, out = run_python_sandboxed("print(")
    assert not ok and ("SyntaxError" in out or "Error" in out)


def test_runtime_error_handled():
    ok, out = run_python_sandboxed("x = 1 / 0\nprint(x)")
    assert not ok and "ZeroDivisionError" in out


def test_no_output_hint():
    ok, out = run_python_sandboxed("y = 2 + 2")  # нет print
    assert ok and "print" in out.lower()


def test_tool_builds_and_runs():
    import asyncio
    t = make_compute_tool()
    assert t.name == "python_exec"
    # тул теперь async (coroutine) — гейт python_exec умеет await HITL при недоверенном контенте.
    # Без taint (чистый прогон) выполняется напрямую.
    res = asyncio.run(t.ainvoke({"code": "print(sum(range(101)))"}))
    assert res.strip() == "5050"


def test_timeout_kills_hang():
    ok, out = run_python_sandboxed("while True: pass", timeout=3)
    assert not ok and ("таймаут" in out or "завис" in out)
