"""Credit assignment: дифференциальная вина по активациям нод (без LLM)."""
from src.improve.graph_learn import OPTIMIZABLE, _node_rates


def test_node_rates_counts_optimizable_only():
    runs = {
        "r1": ["recall", "goal", "reflexion", "decompose", "step_executor"],
        "r2": ["recall", "goal", "reflexion", "fast_answer"],
    }
    rates = _node_rates(runs)
    assert rates["goal"] == 1.0          # активна в обоих
    assert rates["decompose"] == 0.5     # только в одном
    assert "recall" not in rates         # не оптимизируемая — не считается


def test_differential_blame_logic():
    """Нода, активная и в успехах и в неудачах, не должна получать вину."""
    fail_rates = _node_rates({"f1": ["goal", "decompose"], "f2": ["goal", "decompose"]})
    suc_rates = _node_rates({"s1": ["goal", "fast_answer"]})
    blame = {n: fail_rates.get(n, 0.0) - suc_rates.get(n, 0.0) for n in OPTIMIZABLE}
    blame = {n: v for n, v in blame.items() if v > 0}
    assert blame == {"decompose": 1.0}   # goal сократился (1.0 − 1.0), виноват только decompose
