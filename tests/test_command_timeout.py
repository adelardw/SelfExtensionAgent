import asyncio
import sys

from src.coding.harness import Workspace
from src.runtime import hitl


def test_timeout_preserves_bounded_tail_command_and_full_log(tmp_path):
    async def run():
        ws = Workspace(tmp_path, tmp_path/'.sea', {})
        result = await ws.command([sys.executable, '-u', '-c',
            'import time; print("x"*30000); print("entered slow stage"); time.sleep(60)'], timeout=.5)
        assert not result['ok'] and result['timed_out'] and result['exit_code'] == -1
        assert 'entered slow stage' in result['output'] and 'partial' in result['output']
        assert len(result['output']) < 24200 and result['command']
        assert (ws.directory/(result['log_id']+'.log')).stat().st_size > 30000
    asyncio.run(run())


def test_hanging_python_experiment_reports_its_real_line_and_partial_output(tmp_path, monkeypatch):
    from src.coding import worktools
    monkeypatch.setattr(worktools, 'PYTHON_EXPERIMENT_TIMEOUT', 1)
    hitl.set_work_mode('auto')
    async def run():
        ws = Workspace(tmp_path, tmp_path/'.sea', {})
        result = await ws.execute('experiment', {'code': 'print("before loop")\nwhile True:\n    pass\n'})
        assert not result['ok'] and result['timed_out']
        assert 'before loop' in result['output']
        assert 'experiment.py", line' in result['output'] and 'Timeout' in result['output']
        assert 'NOT A GENERAL' in result['output']
        success = await ws.execute('experiment', {'code': 'import sys\nassert len(sys.argv)==1\nprint(42)\n'})
        assert success['ok'] and '42' in success['output'] and 'Timeout' not in success['output']
        (tmp_path/'local_module.py').write_text('VALUE=17\n')
        imported=await ws.execute('experiment',{'code':'from local_module import VALUE\nprint(VALUE)\n'})
        assert imported['ok'] and '17' in imported['output']
    asyncio.run(run())
