"""Actual subprocess confinement, separate from path validation in file tools."""
import asyncio
import json
import platform
import shutil
import sys

import pytest

from src.coding.harness import Workspace
from src.runtime import hitl

pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or not shutil.which('sandbox-exec'),
                                reason='macOS sandbox integration')


def workspace(tmp_path):
    root = tmp_path/'project'; root.mkdir()
    directory = root/'.sea'/'coding'/'test'; directory.mkdir(parents=True)
    return Workspace(root,directory,{})


def test_command_cannot_write_source_or_read_sibling_secret_and_metadata(tmp_path):
    ws = workspace(tmp_path)
    (ws.root/'source.txt').write_text('keep')
    (tmp_path/'private.txt').write_text('private')
    (ws.root/'.env').write_text('API_KEY=private')
    (ws.directory/'state.json').write_text('private state')
    code = '''
import pathlib,json
paths=['source.txt','../private.txt','.env','.sea/coding/test/state.json']
results=[]
for i,p in enumerate(paths):
    try:
        if i==0: pathlib.Path(p).write_text('bad')
        else: pathlib.Path(p).read_text()
        results.append('allowed')
    except PermissionError: results.append('denied')
print(json.dumps(results))
'''
    result = asyncio.run(ws.command([sys.executable,'-c',code]))
    assert result['ok'], result['output']
    assert json.loads(result['output'].strip()) == ['denied']*4
    assert (ws.root/'source.txt').read_text() == 'keep'


def test_command_has_writable_scratch_but_no_network_or_key_environment(tmp_path,monkeypatch):
    ws = workspace(tmp_path)
    monkeypatch.setenv('OPENROUTER_API_KEY','must-not-leak')
    code = '''
import socket,os,pathlib,json
p=pathlib.Path(os.environ['SEA_SCRATCH'])/'result'
p.write_text('compiled output')
try:
    socket.socket().bind(('127.0.0.1',0))
    network=True
except OSError: network=False
print(json.dumps([p.read_text(),network,'OPENROUTER_API_KEY' in os.environ]))
'''
    result = asyncio.run(ws.command([sys.executable,'-c',code]))
    assert result['ok'],result['output']
    assert json.loads(result['output'].strip()) == ['compiled output',False,False]


def test_actor_can_find_python_with_installed_artifact_packages(tmp_path):
    ws=workspace(tmp_path)
    overview=asyncio.run(ws.overview())
    executable=overview['runtime']['python']
    result=asyncio.run(ws.command([executable,'-I','-c','import fitz,pptx; print("artifact imports available")']))
    assert result['ok'] and 'artifact imports available' in result['output']
    # A remote backend must never be told to invoke the host interpreter.
    ws.verifier_tools=frozenset({'read','tree','search'})
    ws.revision+=1
    assert 'runtime' not in asyncio.run(ws.overview())


def test_cpp_can_compile_and_run_inside_scratch(tmp_path):
    compiler = shutil.which('clang++')
    if not compiler: pytest.skip('clang++ unavailable')
    ws = workspace(tmp_path)
    (ws.root/'main.cpp').write_text('#include <iostream>\nint main(){std::cout << 42;}\n')
    target = str(ws.scratch/'program')
    result = asyncio.run(ws.command([compiler,'-std=c++17','main.cpp','-o',target]))
    assert result['ok'],result['output']
    result = asyncio.run(ws.command([target]))
    assert result['ok'] and result['output'] == '42'


def test_extra_authority_still_requires_operator_in_full_auto(tmp_path,monkeypatch):
    ws = workspace(tmp_path)
    called=[]
    async def deny(text): called.append(text); return False
    monkeypatch.setattr(hitl,'confirm',deny)
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})
    result = asyncio.run(ws.command([sys.executable,'-c','print(1)'],project_write=True,network=True))
    assert result['denied'] and called


def test_secret_filter_covers_nested_and_case_variants(tmp_path):
    ws=workspace(tmp_path)
    nested=ws.root/'nested';nested.mkdir()
    for name in ['.env.local','Service.KEY','.npmrc','credentials.json']:
        (nested/name).write_text('dummy')
    code='''
from pathlib import Path
results=[]
for p in Path('nested').iterdir():
    try: p.read_text(); results.append('allowed')
    except PermissionError: results.append('denied')
print(','.join(results))
'''
    result=asyncio.run(ws.command([sys.executable,'-c',code]))
    assert result['ok'] and result['output'].strip()==','.join(['denied']*4)
