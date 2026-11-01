"""Original tests and external edits cannot be relabeled as agent-authored work."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from src.coding.control import SessionControl
from src.coding.harness import Workspace
from src.coding.test_provenance import TestProvenance,originals_preserved
from src.runtime import hitl


@pytest.fixture(autouse=True)
def permissions(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'manual'})


ORIGINAL='''import unittest
class Tests(unittest.TestCase):
    def test_original(self): self.assertEqual(1,1)
'''
ADDED='    def test_added(self): self.assertEqual(2,3)\n'


def setup(tmp_path,**limits):
    directory=tmp_path/'.sea';directory.mkdir()
    (tmp_path/'test_cases.py').write_text(ORIGINAL)
    ws=Workspace(tmp_path,directory,{})
    ws.orchestrator=SimpleNamespace(control=SessionControl())
    ws.test_provenance=TestProvenance.start(ws,**limits)
    hitl.set_work_mode('auto')
    return ws


def edit(ws,old,new):
    assert asyncio.run(ws.execute('edit',{'path':'test_cases.py','old':old,'new':new}))['ok']


def eligible(ws,name='test_added',version=0):
    return ws.test_provenance.eligible_original(ws,'test_cases.py','Tests',name,version)


def test_history_captures_working_copy_and_survives_resume(tmp_path):
    ws=setup(tmp_path)
    edit(ws,ORIGINAL,ORIGINAL+ADDED)
    assert eligible(ws)==ORIGINAL
    assert eligible(ws,'test_original') is None
    restored=TestProvenance.start(ws,resume=True)
    assert restored.eligible_original(ws,'test_cases.py','Tests','test_added',0)==ORIGINAL
    assert restored.original('test_cases.py')==ORIGINAL


def test_external_new_method_is_not_claimed_by_an_unrelated_agent_edit(tmp_path):
    ws=setup(tmp_path)
    (tmp_path/'test_cases.py').write_text(ORIGINAL+ADDED)
    edit(ws,ORIGINAL,ORIGINAL+'    def test_owned(self): self.assertTrue(True)\n')
    assert eligible(ws) is None
    assert eligible(ws,'test_owned')==ORIGINAL


def test_external_edit_revokes_ownership_even_after_another_agent_edit(tmp_path):
    ws=setup(tmp_path);edit(ws,ORIGINAL,ORIGINAL+ADDED)
    (tmp_path/'test_cases.py').write_text(ORIGINAL+ADDED+'# external change\n')
    assert eligible(ws) is None
    edit(ws,'assertEqual(2,3)','assertEqual(2,4)')
    assert eligible(ws) is None


def test_changed_original_assertion_disables_automatic_repair(tmp_path):
    ws=setup(tmp_path);edit(ws,ORIGINAL,ORIGINAL+ADDED)
    edit(ws,'assertEqual(1,1)','assertEqual(1,2)')
    assert eligible(ws) is None


def test_new_operator_scope_cannot_reuse_old_authorship_approval(tmp_path):
    ws=setup(tmp_path);edit(ws,ORIGINAL,ORIGINAL+ADDED)
    ws.orchestrator.control.post('Keep the tests as they are',interrupt=False)
    assert eligible(ws,version=1) is None


@pytest.mark.parametrize('corrupt',[False,True])
def test_missing_or_corrupt_resume_history_is_not_reconstructed(tmp_path,corrupt):
    ws=setup(tmp_path)
    ws.test_provenance.path.write_text('[]') if corrupt else ws.test_provenance.path.unlink()
    assert TestProvenance.start(ws,resume=True) is None


def test_files_omitted_by_size_limit_are_unknown_not_empty(tmp_path):
    ws=setup(tmp_path,file_limit=1)
    edit(ws,ORIGINAL,ORIGINAL+ADDED)
    assert eligible(ws) is None and ws.test_provenance.original('test_cases.py') is None


def test_incomplete_inventory_cannot_authorize_unknown_files(tmp_path):
    ws=setup(tmp_path);ws.test_provenance.data['complete']=False
    assert asyncio.run(ws.execute('write',{'path':'test_new.py','content':ORIGINAL+ADDED}))['ok']
    assert ws.test_provenance.eligible_original(ws,'test_new.py','Tests','test_added',0) is None


@pytest.mark.parametrize('changed',[
    ORIGINAL.replace('assertEqual(1,1)','assertTrue(True)'),
    ORIGINAL+'    def test_original(self): pass\n',
    ORIGINAL.replace('unittest.TestCase','CustomBase'),
    ORIGINAL.replace('import unittest','from fake import unittest'),
])
def test_original_ast_cannot_be_weakened_or_shadowed(changed):
    assert not originals_preserved(ORIGINAL,changed)
