"""Bounded initial source observations, with no model calls or extra authority."""
from __future__ import annotations

import asyncio
import json
from pathlib import PurePosixPath
import re
import time

from .progress import source_window

TEXT_SUFFIXES={'.py','.pyi','.c','.cc','.cpp','.h','.hpp','.rs','.go','.js','.jsx',
    '.ts','.tsx','.java','.kt','.rb','.sh','.json','.toml','.yaml','.yml','.md','.txt',
    '.lean','.v','.smt2','.tex','.sql','.html','.css'}
TEXT_NAMES={'Makefile','Dockerfile','CMakeLists.txt'}


async def prefetch_sources(ws,task,max_bytes,*,permitted,existing=(),record=None,cancelled=lambda:False):
    """Read task-named files, or a small complete inventory, within one byte cap.

    A tree result is only a sample in large projects. No relevance/completeness
    claim is made; search/read remain available. No source is executed here.
    """
    budget=min(4800,max(0,int(max_bytes)))
    result={'reads':{},'hashes':{},'records':[]}
    if budget<160 or not {'tree','read'}<=set(permitted) or cancelled():return result
    record=record or ws.record
    deadline=time.monotonic()+3
    try:tree=await asyncio.wait_for(ws.execute('tree',{},readonly=True),3)
    except (TimeoutError,ValueError,OSError):return result
    if not tree.get('ok'):return result
    inventory=list(dict.fromkeys(str(tree.get('output','')).splitlines()))[:250]
    mentions=set(re.findall(r'[\w./-]+\.[A-Za-z0-9]{1,8}',task))
    basenames={PurePosixPath(p).name for p in mentions}
    mentioned=lambda p:p in mentions or PurePosixPath(p).name in basenames
    candidates=[p for p in inventory if p not in existing and
        (PurePosixPath(p).suffix.lower() in TEXT_SUFFIXES or PurePosixPath(p).name in TEXT_NAMES)
        and (len(inventory)<=8 or mentioned(p))]
    candidates.sort(key=lambda p:not mentioned(p))
    for path in candidates[:8]:
        if cancelled() or time.monotonic()>=deadline:break
        remaining=budget-len(json.dumps(result['reads'],ensure_ascii=False).encode())
        if remaining<160:break
        args={'path':path,'start':1,'end':200}
        try:
            before=ws.file_digest(path)
            observation=await asyncio.wait_for(ws.execute('read',args,readonly=True),deadline-time.monotonic())
            if not observation.get('ok'):continue
            after=ws.file_digest(path)
            if not after or before is not None and before!=after:continue
        except (ValueError,OSError,UnicodeError,TimeoutError):continue
        excerpt=source_window({path:str(observation.get('output',''))},max_bytes=min(2200,remaining)).get(path)
        if not excerpt:continue
        candidate={**result['reads'],path:excerpt}
        if len(json.dumps(candidate,ensure_ascii=False).encode())>budget:continue
        evidence_id=record({**observation,'origin':'runtime_source_prefetch',
            'tool':'read','args':args,'source_hashes':{path:after}})
        result['reads'][path]=excerpt;result['hashes'][path]=after
        result['records'].append({'tool':'read','args':args,'ok':True,'output':excerpt,
            'evidence_id':evidence_id,'origin':'runtime_source_prefetch'})
    return result
