"""Bounded current source coverage for the completion critic.

Source observations are attributable snapshots, never claims of correctness.
Small files cannot starve code later in alphabetical order. Every omission is
explicit; callers must still enforce checks, source freshness and review gates.
"""
from __future__ import annotations
import json


def recent_observations(ws, evidence_ids, limit=12):
    """Select ordinary results before applying the limit; snapshots aren't results."""
    records=[];seen=set()
    for value in reversed(evidence_ids[-1024:]):
        eid=str(value)
        if eid in seen or not eid.isascii() or not eid.isdigit():continue
        seen.add(eid);path=ws.directory/(eid+'.json')
        try:
            if path.is_symlink() or path.stat().st_size>2*1024**2:continue
            item=json.loads(path.read_text())
            if not isinstance(item,dict) or item.get('verifier') or item.get('critic_source'):continue
        except (OSError,ValueError):continue
        records.append((eid,item))
        if len(records)>=limit:break
    return list(reversed(records))


def excerpt(text, allowance, *, captured_all=True):
    if len(text)<=allowance and captured_all:
        return text,False
    marker='\n[OMITTED SOURCE; retrieve missing ranges before supporting affected claims]\n'
    if not captured_all:
        marker='\n[REMAINING SOURCE NOT CAPTURED; retrieve missing ranges]\n'
    if allowance<=len(marker):return marker[:allowance],True
    available=allowance-len(marker)
    if not captured_all:return text[:available]+marker,True
    head=available//2
    return text[:head]+marker+text[-(available-head):],True


async def current_sources(ws, names, *, char_budget=24000, file_limit=32):
    """Water-fill the capacity across files, reallocating unused short-file space."""
    char_budget=max(2000,min(128000,int(char_budget)))
    file_limit=max(1,min(64,int(file_limit)))
    names=list(dict.fromkeys(names))
    captured={};manifest=[]
    for name in names[:file_limit]:
        try:
            before=ws.file_digest(name)
            capture_limit=128000
            text=await ws.source_text(name,capture_limit+1)
            after=ws.file_digest(name)
            if before!=after:
                manifest.append({'path':name,'status':'changed_during_capture'})
                continue
            captured[name]=(text,before,len(text)<=capture_limit)
        except (OSError,ValueError,UnicodeError):
            manifest.append({'path':name,'status':'unavailable'})
    remaining=char_budget;allowances={}
    ordered=sorted(captured,key=lambda n:len(captured[n][0]))
    for i,name in enumerate(ordered):
        quota=min(len(captured[name][0]),remaining//(len(ordered)-i))
        allowances[name]=quota;remaining-=quota
    sources={}
    for name,(text,sha,complete) in captured.items():
        shown,truncated=excerpt(text,allowances[name],captured_all=complete)
        eid=ws.record({'ok':True,'output':shown,'source':name,'source_sha256':sha,
                       'critic_source':True,'truncated':truncated})
        sources[name]={'evidence_id':eid,'text':shown,'sha256':sha,'truncated':truncated}
        manifest.append({'path':name,'sha256':sha,'evidence_id':eid,
                         'status':'excerpt' if truncated else 'complete'})
    return sources,{'files':manifest,'omitted_paths':names[file_limit:file_limit+32],
                    'omitted_file_count':max(0,len(names)-file_limit),
                    'source_chars':sum(len(x['text']) for x in sources.values()),
                    'char_budget':char_budget}
