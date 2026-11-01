"""Bounded exploration: repeated unchanged proposals temporarily leave the action menu."""
import json
from collections import Counter


class FailureFocus:
    """Keep the latest observed failure reachable after the rolling context moves on.

    This is an observation, never a diagnosis or a replacement for verification.
    A changed source snapshot marks it historical rather than silently clearing it.
    """
    def __init__(self, ws, owner='main'):
        if not owner.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Invalid failure-focus owner')
        self.path = ws.directory / f'failure-focus-{owner}.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def observe(self, ws, tool, args, result, evidence_id, steering_version, files=()):
        from .obligations import digest
        from .harness import _atomic
        operation = digest([tool, args])
        if result.get('ok') is False:
            hashes = {}
            for name in sorted(set(files) | ws.changed)[:16]:
                try: hashes[name] = ws.file_digest(name)
                except (OSError, ValueError): hashes[name] = None
            self.state = {'tool': tool, 'operation': operation,
                'args': {k: v if len(str(v)) <= 300 else str(v)[:300]+'…' for k,v in args.items()},
                'evidence_id': str(evidence_id), 'output': str(result.get('output',''))[-2400:],
                'revision': ws.revision, 'steering_version': steering_version, 'source_hashes': hashes}
        elif self.state.get('operation') == operation:
            self.state = {}
        else:
            return
        _atomic(self.path, json.dumps(self.state, ensure_ascii=False))

    def packet(self, ws, steering_version):
        if not self.state or self.state['steering_version'] != steering_version:
            return {}
        stale = self.state['revision'] != ws.revision
        for name, expected in self.state['source_hashes'].items():
            try: stale |= ws.file_digest(name) != expected
            except (OSError, ValueError): stale = True
        return {k:self.state[k] for k in ('tool','args','evidence_id','output')} | {
            'source_changed_since_failure': stale,
            'status': 'historical_failure_requires_recheck' if stale else 'observed_failure',
            'next_step': ('Source changed: run the relevant check after the correction.' if stale else
                          'Use this observed failure. Inspect only missing information or run a discriminating experiment; then correct the actual defect. An unchanged rerun does not diagnose it.')}


async def source_snapshot(ws, name, limit=2200):
    """Refresh known edited source without buying another model read action."""
    try:
        before = ws.file_digest(name)
        text = await ws.source_text(name, limit+1)
        after = ws.file_digest(name)
        if before != after: return None
        suffix = '\n[excerpt truncated; use read for missing ranges]' if len(text)>limit else '\n[EOF]'
        return f'CURRENT SOURCE {name}\n'+text[:limit]+suffix, after
    except (OSError, ValueError):
        return None


def source_window(reads, max_bytes=4800, max_files=8):
    """Keep current observed source by byte capacity, not a two-file cutoff.

    Callers remove stale sources. Values remain observations, not summaries. The
    returned order lets subsequent context fitting evict older sources first.
    """
    selected=[]
    remaining=max_bytes-2
    for name,source in reversed(list(reads.items())):
        size=len(json.dumps({name:source},ensure_ascii=False).encode())
        if size>remaining and not selected:
            marker='\n[excerpt truncated; use read for missing ranges]'
            # JSON encoding (quotes, escaped newlines, Unicode) is the actual cost.
            left,right=0,len(source)
            while left<right:
                middle=(left+right+1)//2
                if len(json.dumps({name:source[:middle]+marker},ensure_ascii=False).encode())<=remaining:
                    left=middle
                else:right=middle-1
            if left:
                source=source[:left]+marker
                size=len(json.dumps({name:source},ensure_ascii=False).encode())
        if size>remaining:continue
        selected.append((name,source));remaining-=size
        if len(selected)>=max_files:break
    return dict(reversed(selected))


def call_signature(tool, args):
    """Identify an operation independently of the order of its JSON keys."""
    return json.dumps([tool, args], sort_keys=True, ensure_ascii=False)


class ActionFrontier:
    def __init__(self):
        self.epoch=None
        self.counts=Counter()
        self.actions=set()
        self.calls={}

    def sync(self,epoch):
        if self.epoch!=epoch:
            self.epoch=epoch;self.counts.clear();self.actions.clear();self.calls.clear()

    def allowed(self,turn):
        return turn.action not in self.actions and (turn.action!='tool' or
            call_signature(turn.tool,turn.args) not in self.calls)

    def note(self,turn):
        value=(turn.tool,turn.args) if turn.action=='tool' else turn.summary if turn.action=='finish' else turn.task if turn.action=='delegate' else 'plan'
        key=json.dumps([turn.action,value],sort_keys=True,ensure_ascii=False)
        self.counts[key]+=1
        if self.counts[key]>=2:
            if turn.action=='tool':
                signature=call_signature(turn.tool,turn.args)
                tool,args=json.loads(signature)
                self.calls[signature]={'tool':tool,'args':args}
            else:
                self.actions.add(turn.action)

    def packet(self):
        if not self.actions and not self.calls:return {}
        return {'blocked_actions':sorted(self.actions),'blocked_calls':list(self.calls.values()),
                'reason':'Repeated unchanged proposals. A blocked call excludes only that tool with those exact arguments; corrected arguments remain available. New observations/source changes/operator steering reset this temporary restriction.'}
