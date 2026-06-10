"""Кроссплатформенность device-навыков + syscall-песочница + vision-склейка."""
import importlib.util
import sys

import pytest


def _load_device():
    spec = importlib.util.spec_from_file_location(
        "device_control_test", "src/skills/device_control/device_control.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["device_control_test"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def dc(monkeypatch):
    monkeypatch.setenv("AGENT_DRY_RUN", "1")
    return _load_device()


def test_open_url_each_os(dc, monkeypatch):
    captured = {}

    def fake_run(cmd, timeout=20):
        captured["cmd"] = cmd
        return True, "[dry]"

    monkeypatch.setattr(dc, "_run", fake_run)
    for os_name, expect in [("Darwin", "open"), ("Windows", "start"), ("Linux", None)]:
        monkeypatch.setattr(dc, "_OS", os_name)
        if os_name == "Linux":
            monkeypatch.setattr(dc, "_first_tool", lambda *a: "xdg-open")
        r = dc.open_url.invoke({"url": "example.com"})
        assert "https://example.com" in r
        if expect:
            assert expect in " ".join(captured["cmd"])


def test_macos_only_ops_degrade_gracefully(dc, monkeypatch):
    monkeypatch.setattr(dc, "_OS", "Linux")
    out = dc.type_text.invoke({"text": "hi"})
    assert "только на macOS" in out
    out = dc.scroll.invoke({"direction": "down"})
    assert "только на macOS" in out


def test_capture_screen_linux_needs_tool(dc, monkeypatch):
    monkeypatch.setattr(dc, "_OS", "Linux")
    monkeypatch.setattr(dc, "_first_tool", lambda *a: "")  # ни одной утилиты
    monkeypatch.delenv("AGENT_DRY_RUN", raising=False)
    out = dc.capture_screen.invoke({})
    assert "grim" in out and "scrot" in out  # подсказка что доставить


def test_analyze_screen_dry_run(dc):
    out = dc.analyze_screen.invoke({"question": "что видно"})
    assert "dry-run" in out


def test_syscall_sandbox_prefix_off(monkeypatch):
    from src import utils

    monkeypatch.setenv("AGENT_SYSCALL_SANDBOX", "0")
    assert utils._syscall_sandbox_prefix() == []


def test_syscall_sandbox_prefix_linux_bwrap(monkeypatch):
    import platform

    from src import utils

    monkeypatch.setenv("AGENT_SYSCALL_SANDBOX", "auto")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(utils.shutil, "which", lambda b: "/usr/bin/bwrap" if b == "bwrap" else None)
    prefix = utils._syscall_sandbox_prefix()
    assert prefix and prefix[0] == "bwrap" and "--ro-bind" in prefix


def test_sandbox_still_works_with_rlimits(tmp_path, monkeypatch):
    """Базовый rlimits-путь не сломан добавлением syscall-обёртки."""
    monkeypatch.setenv("AGENT_SYSCALL_SANDBOX", "0")
    from src.utils import run_tool_sandboxed

    f = tmp_path / "s.py"
    f.write_text(
        "from langchain_core.tools import tool\n"
        "@tool\n"
        "def ping(x: str) -> str:\n    '''p'''\n    return 'pong ' + x\n",
        encoding="utf-8",
    )
    ok, res = run_tool_sandboxed(f, "ping", {"x": "ok"})
    assert ok and "pong ok" in res
