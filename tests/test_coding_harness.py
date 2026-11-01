"""Offline end-to-end coding trajectories: real files/commands, scripted model turns."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from src.coding.harness import CodingHarness, Turn, Workspace
from src.coding.model import LocalModel
from src.runtime import hitl


class Scripted:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    async def __call__(self, messages, schema, effort):
        self.requests.append((messages, effort))
        assert self.turns, "unexpected extra model call"
        return schema.model_validate(self.turns.pop(0))


@pytest.fixture(autouse=True)
def coding_permissions(monkeypatch):
    monkeypatch.setattr(hitl, "_work_mode", {"": "auto"})


def plan():
    return {"action": "plan", "plan": ["Inspect", "Implement", "Verify"]}


def tool(tool_name, **args):
    return {"action": "tool", "tool": tool_name, "args": args}


def finish(summary="done"):
    return {"action": "finish", "summary": summary}


def harness(tmp_path, model, **config):
    return CodingHarness(tmp_path, model=model, config={"max_turns": 20, **config})


def test_repairs_using_real_check_evidence(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    check = [sys.executable, "-B", "-c", "from app import VALUE; assert VALUE == 3, VALUE"]
    model = Scripted(plan(), tool("read", path="app.py"),
        tool("edit", path="app.py", old="VALUE = 1", new="VALUE = 2"),
        tool("check", name="unit"), tool("edit", path="app.py", old="VALUE = 2", new="VALUE = 3"),
        finish("premature"), tool("check", name="unit"), finish())
    events = []
    result = asyncio.run(harness(tmp_path, model, checks={"unit": check}).run("fix", progress=events.append))
    assert result["status"] == "done"
    assert (tmp_path / "app.py").read_text() == "VALUE = 3\n"
    assert result["checks_passed"] == ["unit"]
    assert any(e["phase"] == "repair" for e in events)
    assert any("Completion rejected" in str(messages) for messages, _ in model.requests)
    logs = list(Path(result["checkpoint"]).parent.glob("[1-9]*.json"))
    assert any(json.loads(p.read_text()).get("exit_code") == 1 for p in logs)


def test_check_observation_refers_to_evidence_record_not_subprocess_log_uuid(tmp_path):
    model=Scripted(plan(),tool('check',name='unit'),finish())
    result=asyncio.run(harness(tmp_path,model,checks={'unit':[sys.executable,'-c','print(42)']}).run('Check project'))
    packet=json.loads(model.requests[-1][0][-1]['content'])
    observation=next(x for x in packet['recent'] if x.get('tool')=='check')
    eid=observation['evidence_id']
    assert eid.isdigit()
    record=json.loads((Path(result['checkpoint']).parent/(eid+'.json')).read_text())
    assert record['ok'] and '42' in record['output']
    assert record['log_id']!=eid


def test_plan_mode_cannot_write_or_execute(tmp_path):
    model = Scripted(tool("write", path="evil.py", content="bad"),
                     tool("run", argv=[sys.executable, "-c", "raise Exception"]), plan())
    result = asyncio.run(harness(tmp_path, model).run("plan", execution="plan"))
    assert result["status"] == "planned" and not (tmp_path / "evil.py").exists()
    assert not result["tools_used"]


def test_worker_readonly_enforced_and_no_recursive_delegation(tmp_path):
    model = Scripted(plan(), {"action": "delegate", "role": "review", "task": "Review one module"},
                     tool("write", path="unsafe", content="bad"),
                     {"action": "delegate", "task": "recursive"}, finish("reviewed"), finish())
    result = asyncio.run(harness(tmp_path, model).run("inspect"))
    assert result["delegates"] == 1 and not (tmp_path / "unsafe").exists()
    assert any("read-only worker" in str(m) for m, _ in model.requests)
    assert any("use a read-only" in str(m) for m, _ in model.requests)


def test_delegate_count_is_bounded(tmp_path):
    d = {"action": "delegate", "task": "inspect"}
    model = Scripted(plan(), d, finish("worker"), d, finish())
    result = asyncio.run(harness(tmp_path, model, max_delegates=1).run("inspect"))
    assert result["delegates"] == 1 and len(model.requests) == 5


def test_resume_never_trusts_old_verification(tmp_path):
    one = Scripted(plan(), tool("write", path="a.txt", content="ok"))
    first = asyncio.run(harness(tmp_path, one, max_turns=2, checks={"unit": [sys.executable, "-c", "pass"]}).run("create", session_id="s"))
    assert first["status"] == "paused"
    two = Scripted(finish("false completion"), tool("check", name="unit"), finish())
    result = asyncio.run(harness(tmp_path, two, checks={"unit": [sys.executable, "-c", "pass"]}).run("", session_id="s", resume=True))
    assert result["status"] == "done" and result["changed"] == ["a.txt"]
    assert len(two.requests) == 3
    assert "create" in str(two.requests[0])


def test_unverified_work_is_not_marked_complete(tmp_path):
    model = Scripted(plan(), tool("write", path="a", content="ok"), finish())
    result = asyncio.run(harness(tmp_path, model, checks={}).run("create"))
    assert result["status"] == "paused" and not result["checks_passed"]


def test_command_after_passing_check_invalidates_verification(tmp_path,monkeypatch):
    async def approve(_): return True
    monkeypatch.setattr(hitl,'confirm',approve)
    model = Scripted(plan(), tool("write", path="a", content="1"), tool("check", name="unit"),
        tool("run", argv=[sys.executable, "-c", "from pathlib import Path; Path('a').write_text('2')"],project_write=True),
        finish("premature"), tool("check", name="unit"), finish())
    r = asyncio.run(harness(tmp_path, model, checks={"unit": [sys.executable, "-c", "pass"]}).run("change"))
    assert r["status"] == "done" and len(model.requests) == 7
    assert (tmp_path/'a').read_text()=='2'  # The command really mutated the project.


def test_permissions_are_not_upgraded_by_phase(tmp_path):
    hitl.set_work_mode("plan")
    model = Scripted(plan(), tool("write", path="a", content="bad"), finish())
    result = asyncio.run(harness(tmp_path, model).run("write"))
    assert not (tmp_path / "a").exists()
    assert hitl.is_plan() and result["tools_used"] == ["write"]


def test_denial_stops_without_retry(tmp_path, monkeypatch):
    hitl.set_work_mode("manual")
    async def deny(_):
        return False
    monkeypatch.setattr(hitl, "confirm", deny)
    model = Scripted(plan(), tool("write", path="a", content="bad"))
    result = asyncio.run(harness(tmp_path, model).run("write"))
    assert result["status"] == "paused" and not (tmp_path / "a").exists()


def test_paths_secrets_and_symlinks_are_blocked(tmp_path):
    ws = Workspace(tmp_path, tmp_path / ".sea")
    for name in ("../escape", ".env", ".aws/config", ".git/config", ".sea/coding/state.json"):
        with pytest.raises(ValueError):
            ws.path(name)
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        ws.path("outside/file")


def test_bounded_history_and_exact_reasoning_choice(tmp_path):
    (tmp_path / "large.txt").write_text("x" * 10000)
    model = Scripted(plan(), *[tool("read", path="large.txt") for _ in range(4)], finish())
    result = asyncio.run(harness(tmp_path, model).run("inspect", reasoning="high"))
    assert result["status"] == "done"
    assert all(effort == "high" for _, effort in model.requests)
    assert max(len(str(m)) for m, _ in model.requests) < 22000


def test_auto_reasoning_escalates_on_evidence(tmp_path):
    check = [sys.executable, "-c", "raise SystemExit(1)"]
    model = Scripted(plan(), tool("check", name="unit"), tool("check", name="unit"), finish())
    result = asyncio.run(harness(tmp_path, model, max_turns=4, checks={"unit": check}).run("inspect"))
    assert model.requests[-1][1] == "high"
    # A failure must not be called success even when no edit was made.
    assert result["status"] != "done"


def test_ollama_checks_parameter_count_and_never_uses_cloud(monkeypatch):
    paths = []
    def transport(req):
        paths.append(req.url.path)
        return httpx.Response(200, json={"model_info": {"general.parameter_count": 14e9}})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(transport)))
    model = LocalModel({"model": "oversize"})
    with pytest.raises(ValueError, match="10B"):
        asyncio.run(model([{"role": "user", "content": "hi"}], Turn, "low"))
    assert paths == ["/api/show"]


def test_native_ollama_schema_reasoning_and_usage(monkeypatch):
    payloads = []
    def transport(req):
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"details": {"parameter_size": "8.2B"}, "capabilities": ["thinking"]})
        payloads.append(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(finish())}, "prompt_eval_count": 100, "eval_count": 20})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(transport)))
    model = LocalModel({})
    asyncio.run(model([{"role": "user", "content": json.dumps({"task":"fix", "recent":[{"output":"я" * 20000}]}, ensure_ascii=False)}], Turn, "low"))
    asyncio.run(model([{"role": "user", "content": "fix"}], Turn, "high"))
    assert payloads[0]["think"] is False and payloads[1]["think"] is True
    assert payloads[0]["options"]["num_ctx"] == 16384
    assert payloads[0]["format"]["oneOf"][1]["properties"]["tool"]["enum"]
    assert model.usage()["tokens"] == 240
    assert len(json.dumps(payloads[0]["messages"]).encode()) < 60000


def test_checkpoint_symlink_escape_blocked(tmp_path):
    target = tmp_path.parent / (tmp_path.name + "-other")
    target.mkdir()
    (tmp_path / ".sea").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(harness(tmp_path, Scripted()).run("task"))


def test_workspace_lease_excludes_other_writers(tmp_path):
    from src.coding.locking import workspace_lease
    with workspace_lease(tmp_path):
        with pytest.raises(RuntimeError, match="another coding run"):
            with workspace_lease(tmp_path):
                pass
    with workspace_lease(tmp_path):
        pass


def test_no_checkpoint_is_not_silently_resumed(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint"):
        asyncio.run(harness(tmp_path, Scripted()).run("", resume=True))


def test_only_verified_workflows_enter_library(tmp_path):
    model = Scripted(plan(), tool("write", path="a", content="ok"), tool("check", name="unit"), finish())
    r = asyncio.run(harness(tmp_path, model, checks={"unit": [sys.executable, "-c", "pass"]}).run("create verified file"))
    library = Path(r["checkpoint"]).parent.parent / "lessons.json"
    assert json.loads(library.read_text())[0]["checks"] == ["unit"]
    model = Scripted(plan(), tool("write", path="b", content="ok"), finish())
    asyncio.run(harness(tmp_path, model, checks={}).run("unverified", session_id="other"))
    assert len(json.loads(library.read_text())) == 1


def test_full_evidence_can_be_paged(tmp_path):
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "1.json").write_text(json.dumps({"output": "first\nsecond\nthird"}))
    ws = Workspace(tmp_path, directory)
    result = asyncio.run(ws.execute("evidence", {"id": "1", "start": 2, "end": 2}))
    assert result["output"] == "second"


def test_crashed_mutation_requires_reverification_on_resume(tmp_path):
    model = Scripted(plan(), tool("run", argv=["anything"]))
    h = harness(tmp_path, model)
    original = Workspace.execute
    async def crash(self, name, args, readonly=False):
        if name == "run":
            (tmp_path / "partial").write_text("changed before crash")
            raise RuntimeError("simulated crash")
        return await original(self, name, args, readonly)
    from unittest.mock import patch
    with patch.object(Workspace, "execute", crash), pytest.raises(RuntimeError):
        asyncio.run(h.run("write", session_id="crash"))
    model2 = Scripted(finish())
    result = asyncio.run(harness(tmp_path, model2, checks={}).run("", session_id="crash", resume=True))
    assert result["status"] == "paused" and "проверка не выполнена" in result["final_answer"]


def test_coding_http_dispatch_uses_requested_workspace_and_effort(tmp_path, monkeypatch):
    import src.interface.server as server
    import src.coding.harness as module
    seen = {}
    class FakeHarness:
        def __init__(self, root):
            seen["root"] = root
        async def run(self, query, **kwargs):
            seen.update(kwargs)
            kwargs["progress"]({"phase": "repair", "message": "fix", "usage": {"tokens": 42}})
            return {"status": "done", "final_answer": "fixed", "mode": "coding:verify", "plan": [],
                    "changed": ["a.py"], "checks_passed": ["unit"], "delegates": 1, "tools_used": ["check"], "tokens": 42}
    monkeypatch.setattr(module, "CodingHarness", FakeHarness)
    monkeypatch.setattr(server.chat_store, "record_turn", lambda *a: None)
    monkeypatch.setattr(server.chat_store, "get_thread", lambda *a: {})
    monkeypatch.setattr(server, "_RUNS", {"r": {"steps": [], "status": "running"}})
    inp = server.ChatIn(user_id="test", query="fix", task_mode="coding", reasoning="high", workspace=str(tmp_path))
    asyncio.run(server._run_graph("r", inp, "t"))
    result = server.run_status("r")
    assert result["status"] == "done" and result["answer"] == "fixed"
    assert result["tools_used"] == [{"name": "check", "sec": None}]
    assert seen["root"] == tmp_path and seen["reasoning"] == "high"


def test_exact_registered_argv_is_a_check_not_a_new_mutation(tmp_path):
    check = [sys.executable, "-c", "pass"]
    model = Scripted(plan(), tool("write", path="a", content="ok"), tool("run", argv=check), finish())
    result = asyncio.run(harness(tmp_path, model, checks={"unit": check}).run("create and verify"))
    assert result["status"] == "done" and result["checks_passed"] == ["unit"]
    assert result["tools_used"] == ["write", "check"]
