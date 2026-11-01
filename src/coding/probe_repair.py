"""Bounded repair of a verifier program, without silently discarding its failure.

Eligibility and preservation checks are heuristics, not proof that an oracle is correct.
The semantic reviewer still receives both programs and must resolve the original failure.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel,ConfigDict,Field


class ProbeEdit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    old:str=Field(min_length=1,max_length=6000)
    new:str=Field(max_length=6000)


class ProbeRepair(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id:str=Field(pattern=r'^[0-9]+$')
    diagnosis:Literal['probe_error','candidate_or_unknown']
    reason:str=Field(min_length=10,max_length=600)
    edits:list[ProbeEdit]=Field(max_length=3)


class ProbeRepairs(BaseModel):
    model_config=ConfigDict(extra='forbid')
    repairs:list[ProbeRepair]=Field(max_length=3)


def repairable_failure(observation):
    if observation['tool']!='probe' or observation['ok'] or observation.get('stale'):return False
    # A traceback is a reason to investigate the CHECKER, not a diagnosis. Keep
    # assertion counterexamples on the existing candidate-repair path.
    output=observation.get('output','')
    lines=[line for line in output.splitlines() if re.match(r'^[\w.]+(?:Error|Exception)(?::|$)',line)]
    return ('Traceback (most recent call last)' in output and bool(lines)
            and not lines[-1].startswith('AssertionError'))


def patch_probe(code,edits):
    original=ast.parse(code)
    for edit in edits:
        if not edit.old or code.count(edit.old)!=1:raise ValueError('Probe edit must match one exact source occurrence')
        code=code.replace(edit.old,edit.new,1)
    if len(code)>12000:raise ValueError('Repaired probe exceeds source limit')
    revised=ast.parse(code)
    def protected(tree):
        return Counter(ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                       if isinstance(n,(ast.Assert,ast.Import,ast.ImportFrom)))
    before,after=protected(original),protected(revised)
    if any(after[n]<count for n,count in before.items()):
        raise ValueError('Probe repair cannot remove or alter existing assertions or imports')
    if ast.dump(original)==ast.dump(revised):raise ValueError('Probe repair must change executable code')
    return code
