"""Session-local source history for narrowly scoped repair of agent-added tests."""
from __future__ import annotations
import ast
import copy
import hashlib
import json


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def methods(source):
    tree=ast.parse(source)
    return {(cls.name,node.name) for cls in tree.body if isinstance(cls,ast.ClassDef)
        for node in cls.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}


def originals_preserved(original,current):
    """Retain original statements/methods; this is source protection, not a proof."""
    before=ast.parse(original);after=ast.parse(current)
    dump=lambda node:ast.dump(node,include_attributes=False)
    for node in before.body:
        if not isinstance(node,ast.ClassDef):
            if dump(node) not in {dump(n) for n in after.body}:return False
            continue
        matches=[n for n in after.body if isinstance(n,ast.ClassDef) and n.name==node.name]
        if len(matches)!=1:return False
        new=matches[0]
        old_shell=copy.deepcopy(node);new_shell=copy.deepcopy(new)
        old_shell.body=new_shell.body=[]
        if dump(old_shell)!=dump(new_shell):return False
        if any(dump(n) not in {dump(m) for m in new.body} for n in node.body):return False
        names=[n.name for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        if len(names)!=len(set(names)):return False
    return True


class TestProvenance:
    __test__=False

    def __init__(self,path,data):
        self.path,self.data=path,data

    @classmethod
    def start(cls,ws,resume=False,*,file_limit=131072,total_limit=4000000):
        path=ws.directory/'test-provenance.json'
        if resume:return cls.restore(ws,path)
        paths=list(ws.files());files={};used=0
        for name in paths:
            if not name.endswith('.py'):continue
            try:
                source=ws.path(name);size=source.stat().st_size
                if size>file_limit or used+size>total_limit:continue
                text=source.read_text();used+=size
                files[name]={'source':text,'digest':digest(text)}
            except (OSError,UnicodeError,ValueError):continue
        data={'version':1,'root':str(ws.root),'complete':getattr(ws,'inventory_complete',False),
              'existing':paths,'files':files,'owned':{}}
        result=cls(path,data);result.save();return result

    @classmethod
    def restore(cls,ws,path):
        try:
            data=json.loads(path.read_text())
            if not isinstance(data,dict) or data.get('version')!=1 or data.get('root')!=str(ws.root):return None
            if not isinstance(data.get('existing'),list) or not isinstance(data.get('files'),dict) or not isinstance(data.get('owned'),dict):return None
            return cls(path,data)
        except (OSError,ValueError,TypeError):return None

    def save(self):
        from .harness import _atomic
        _atomic(self.path,json.dumps(self.data,ensure_ascii=False))

    def fork(self,ws,owner):
        if not owner.replace('-','').isalnum():raise ValueError('Invalid provenance owner')
        result=TestProvenance(ws.directory/('test-provenance-'+owner+'.json'),copy.deepcopy(self.data))
        result.data['root']=str(ws.root);result.save();return result

    def original(self,name):
        if name not in self.data['existing']:
            return '' if self.data.get('complete') else None
        value=self.data['files'].get(name)
        if not isinstance(value,dict) or not isinstance(value.get('source'),str):return None
        return value['source'] if digest(value['source'])==value.get('digest') else None

    def note_edit(self,ws,name,before,after):
        if not name.endswith('.py'):return
        owned=self.data['owned'];previous=owned.pop(name,{})
        try:
            introduced=methods(after)-methods(before)
            retained={tuple(m) for m in previous.get('methods',[])} if previous.get('digest')==digest(before) else set()
            authored=(introduced|retained)&methods(after)
            original=self.original(name)
            if original is None:return
            authored-=methods(original)
            control=getattr(getattr(ws,'orchestrator',None),'control',None)
            version=control.version(getattr(ws,'owner_id','main')) if control else 0
            owned[name]={'digest':digest(after),'methods':sorted(authored),'steering_version':version}
        except (SyntaxError,ValueError,TypeError):pass
        finally:self.save()

    def eligible_original(self,ws,name,class_name,method_name,steering_version):
        original=self.original(name);owned=self.data['owned'].get(name,{})
        if original is None or [class_name,method_name] not in [list(m) for m in owned.get('methods',[])]:return None
        if owned.get('steering_version')!=steering_version:return None
        try:
            current=ws.path(name).read_text()
            if digest(current)!=owned.get('digest') or not originals_preserved(original,current):return None
        except (OSError,ValueError,SyntaxError):return None
        return original
