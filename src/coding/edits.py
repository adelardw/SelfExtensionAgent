"""Bounded coherent edits through the existing per-file execution boundary.

This is sequential, not a filesystem transaction. Partial application is explicit;
each backend retains its permissions, exact-match checks and revision accounting.
"""
from __future__ import annotations

import json


async def execute_batch(ws, args, *, readonly=False):
    changes=args.get('changes')
    if set(args)!={'changes'} or not isinstance(changes,list) or not 1<=len(changes)<=8:
        return {'ok':False,'output':'Use edit {changes:[{path,old,new},...]} with 1..8 replacements, or the single-file form. Do not mix forms.'}
    for change in changes:
        if (not isinstance(change,dict) or set(change)!={'path','old','new'}
                or any(not isinstance(v,str) for v in change.values())
                or not change['path'] or not change['old']):
            return {'ok':False,'output':'Each replacement needs exactly path, old and new strings; path and old must be nonempty. No replacements applied.'}
    if len(json.dumps(changes,ensure_ascii=False).encode())>100000:
        return {'ok':False,'output':'Combined edit exceeds 100000 UTF-8 bytes. No replacements applied.'}
    applied=[]
    for index,change in enumerate(changes):
        # Deliberately re-enter the normal tool boundary. Never write files or
        # waive a confirmation here, including in a remote container backend.
        result=await ws.execute('edit',change,readonly=readonly)
        if not result.get('ok'):
            return {**result,'ok':False,'changed_paths':list(dict.fromkeys(applied)),
                    'completed_replacements':index,'failed_index':index,
                    'output':f'Replacement {index+1}/{len(changes)} failed. '
                        f'{index} earlier replacements remain applied; later replacements were not run.\n'
                        +str(result.get('output',''))}
        applied.append(change['path'])
    return {'ok':True,'changed_paths':list(dict.fromkeys(applied)),
            'completed_replacements':len(changes),
            'output':'Applied '+str(len(changes))+' exact replacements in '+', '.join(dict.fromkeys(applied))+
                     '. Run registered checks after the final change.'}
