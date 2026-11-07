"""Execute generated artifacts in a small, networkless, read-only Docker container."""
import asyncio
import json
from pathlib import Path
import uuid

IMAGE='python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54'
DRIVER='''import sys,json,importlib.util
sys.path.insert(0,"/work")
spec=importlib.util.spec_from_file_location("solution","/work/solution.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for line in sys.stdin:
 try: print(json.dumps(m.solve(json.loads(line)),ensure_ascii=False,allow_nan=False),flush=True)
 except Exception as e: print(json.dumps({"__execution_error__":type(e).__name__+": "+str(e)[:400]}),flush=True)
'''


async def run(project, argv, stdin='', timeout=20):
    name='sea-600-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--rm','-i','--name',name,'--network','none','--read-only',
         '--cpus','0.5','--memory','192m','--pids-limit','64','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--user','65534:65534',
         '--tmpfs','/tmp:rw,noexec,nosuid,size=16m','--mount',f'type=bind,src={project.resolve()},dst=/work,readonly',
         '-w','/work','-e','PYTHONDONTWRITEBYTECODE=1',IMAGE,*argv]
    proc=await asyncio.create_subprocess_exec(*cmd,stdin=asyncio.subprocess.PIPE,
             stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    async def bounded(stream):
        out=bytearray()
        while chunk:=await stream.read(8192):
            out.extend(chunk)
            if len(out)>131072:raise ValueError('sandbox output limit exceeded')
        return out.decode(errors='replace')
    async def feed():
        try:
            proc.stdin.write(stdin.encode());await proc.stdin.drain()
        except (BrokenPipeError,ConnectionResetError):pass
        finally:proc.stdin.close()
    tasks=[asyncio.create_task(bounded(proc.stdout)),asyncio.create_task(bounded(proc.stderr)),asyncio.create_task(feed())]
    try:
        values=await asyncio.wait_for(asyncio.gather(*tasks),timeout)
        await asyncio.wait_for(proc.wait(),3)
        return {'exit_code':proc.returncode,'stdout':values[0],'stderr':values[1]}
    except (asyncio.TimeoutError,ValueError) as exc:
        return {'exit_code':-1,'stdout':'','stderr':type(exc).__name__+': '+str(exc)}
    finally:
        if proc.returncode is None:proc.kill()
        await proc.wait()
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        cleanup=await asyncio.create_subprocess_exec('docker','rm','-f',name,
            stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
        await cleanup.wait()


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


async def grade(project, cases, kind):
    argv=['python','-I','-c',DRIVER] if kind=='code' else ['python','cli.py']
    result=await run(project,argv,'\n'.join(canonical(c['input']) for c in cases)+'\n')
    try:
        values=[json.loads(line) for line in result['stdout'].splitlines()]
        matches=[i<len(values) and canonical(values[i])==canonical(c['output']) for i,c in enumerate(cases)]
    except (ValueError,TypeError):values=[];matches=[False]*len(cases)
    ok=result['exit_code']==0 and len(values)==len(cases) and all(matches)
    first=next((i for i,matched in enumerate(matches) if not matched),None)
    return {'passed':ok,'matched':sum(matches),'checks':len(cases),'process':result,
            'first_failure':({'input':cases[first]['input'],'expected':cases[first]['output'],
                              'actual':values[first] if first<len(values) else None} if first is not None else None)}


def write_files(project,files):
    if not isinstance(files,list) or not 1<=len(files)<=24:raise ValueError('expected 1..24 files')
    paths=set();total=0
    for file in files:
        if set(file)!={'path','content'} or not isinstance(file['content'],str):raise ValueError('invalid file')
        path=Path(file['path'])
        if path.is_absolute() or any(p in {'..','.git','.sea','.env'} or p.startswith('.') for p in path.parts):
            raise ValueError('unsafe artifact path')
        if not path.parts or str(path) in paths:raise ValueError('empty or duplicate artifact path')
        if any(Path(p) in path.parents or path in Path(p).parents for p in paths):raise ValueError('conflicting file/directory paths')
        if path.suffix not in {'.py','.md','.toml','.txt','.json'}:raise ValueError('unsupported artifact type')
        total+=len(file['content'].encode());paths.add(str(path))
        if total>250000:raise ValueError('artifact size limit exceeded')
    project.mkdir(parents=True,exist_ok=False)
    for file in files:
        path=project/file['path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(file['content']);path.chmod(0o644)
    return sorted(paths)
