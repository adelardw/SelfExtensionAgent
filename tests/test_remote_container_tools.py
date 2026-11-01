"""Exercise the exact container helper in a temporary root, without Docker or API calls."""
import base64
import json
from pathlib import Path
import subprocess
import sys

PROGRAM=(Path(__file__).resolve().parents[1]/'src'/'eval'/'remote_files.py').read_text()


def invoke(root,tool,args):
    program=PROGRAM
    payload=base64.b64encode(json.dumps([tool,args,str(root)]).encode()).decode()
    result=subprocess.run([sys.executable,'-I','-c',program,payload],capture_output=True,text=True,timeout=5)
    return result,json.loads(result.stdout) if result.returncode==0 else None


def test_remote_file_tool_preserves_source_mode_and_returns_hash(tmp_path):
    p=tmp_path/'worker.py';p.write_text('VALUE=1\n');p.chmod(0o755)
    result,data=invoke(tmp_path,'edit',{'path':'worker.py','old':'VALUE=1','new':'VALUE=2'})
    assert result.returncode==0 and data['ok'] and data['source_hash']
    assert p.read_text()=='VALUE=2\n' and p.stat().st_mode&0o777==0o755
    _,data=invoke(tmp_path,'read',{'path':'worker.py'})
    assert 'VALUE=2' in data['output'] and 'of 1' in data['output']


def test_remote_tools_cannot_read_outside_root_or_overwrite_existing_file(tmp_path):
    root=tmp_path/'project';root.mkdir()
    (tmp_path/'private.txt').write_text('must-not-appear')
    result,_=invoke(root,'read',{'path':'../private.txt'})
    assert result.returncode!=0 and 'must-not-appear' not in result.stdout+result.stderr
    (root/'file.py').write_text('original')
    result,_=invoke(root,'write',{'path':'file.py','content':'replaced'})
    assert result.returncode!=0 and (root/'file.py').read_text()=='original'
