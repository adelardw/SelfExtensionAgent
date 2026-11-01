"""Cheap format checks on delivered files; generation in a scratch copy is not delivery."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Executed with -I, in the existing command sandbox, not in the agent process.
PROBE = r'''
import json,sys,zipfile
from pathlib import Path
p=Path(sys.argv[1]); kind=p.suffix.lower()
try:
    if p.stat().st_size>32*1024*1024: raise ValueError('Artifact exceeds 32 MiB inspection limit; use a targeted independent check')
    result={'ok':True,'format':kind,'bytes':p.stat().st_size}
    if kind=='.pdf':
        import fitz
        with fitz.open(p) as doc:
            if doc.needs_pass or not doc.page_count: raise ValueError('PDF is encrypted or has no pages')
            result.update(pages=doc.page_count,text='\n'.join(doc[i].get_text()[:500] for i in range(min(4,doc.page_count))))
    elif kind=='.pptx':
        with zipfile.ZipFile(p) as z:
            if sum(i.file_size for i in z.infolist())>64*1024*1024 or len(z.infolist())>5000:
                raise ValueError('Expanded presentation exceeds inspection limit')
            for name in z.namelist():
                if name.endswith(('.xml','.rels')):
                    raw=z.read(name)
                    if b'<!DOCTYPE' in raw or b'<!ENTITY' in raw: raise ValueError('XML declarations are not allowed')
        from pptx import Presentation
        prs=Presentation(p)
        if not len(prs.slides): raise ValueError('Presentation has no slides')
        # Count only shapes on actual slides, not unreferenced ZIP chart parts.
        shapes=[shape for slide in prs.slides for shape in slide.shapes]
        result.update(slides=len(prs.slides),ratio=prs.slide_width/prs.slide_height,
            charts=sum(shape.has_chart for shape in shapes),tables=sum(shape.has_table for shape in shapes),
            text='\n'.join(shape.text[:500] for shape in shapes if shape.has_text_frame)[:2000])
    elif kind=='.json':
        value=json.loads(p.read_text())
        result.update(json_type=type(value).__name__,items=len(value) if isinstance(value,(list,dict)) else None)
    print(json.dumps(result,ensure_ascii=False))
except Exception as exc:
    print(json.dumps({'ok':False,'error':type(exc).__name__+': '+str(exc)[:500]}))
'''


async def inspect_artifact(ws, path):
    """Metadata is necessary evidence only. Semantic completeness stays with the critic."""
    observations = getattr(ws, 'artifact_observations', None)
    if observations is None:
        observations = ws.artifact_observations = {}
    if not await ws.file_exists(path):
        result = {'ok':False, 'error':'Named deliverable is absent from the actual workspace; scratch/check outputs do not count.'}
    elif Path(path).suffix.lower() not in {'.pdf','.pptx','.json'} or 'probe' not in ws.verifier_tools:
        result = {'ok':True, 'scope':'existence only; content requires independent verification'}
    else:
        native = ws.path(path)
        if native.stat().st_size > 32*1024*1024:
            result = {'ok':False,'error':'Artifact exceeds bounded format inspection limit'}
        else:
            fingerprint = ws.file_digest(path)
            cache = getattr(ws, '_artifact_cache', {})
            if path in cache and cache[path][0] == fingerprint:
                result = cache[path][1]
            else:
                run = await ws.command([sys.executable,'-I','-c',PROBE,str(native)],timeout=10)
                try:
                    result = json.loads(run['output']) if run['ok'] else {'ok':False,'error':run['output'][-600:]}
                    if not isinstance(result,dict) or not isinstance(result.get('ok'),bool): raise ValueError()
                except (ValueError,TypeError):
                    result = {'ok':False,'error':'Format inspector did not produce a valid observation'}
                if ws.file_digest(path) != fingerprint:
                    result = {'ok':False,'error':'Artifact changed during inspection; recheck current file'}
                else:
                    cache[path] = fingerprint, result
                    ws._artifact_cache = cache
    observations[path] = result
    return result
