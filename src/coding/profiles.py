"""Task-created thinking profiles; runtime capabilities remain independently bounded."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from .worktools import TOOLS

ToolName = Literal[tuple(TOOLS)]

BaseMode = Literal['inspect','reason','plan','research','review','repair','implement','verify']


class ProfileSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,47}$')
    base: BaseMode
    strategy: str = Field(min_length=10,max_length=1200)
    tools: list[ToolName] = Field(max_length=32)
    exit_when: list[str] = Field(min_length=1,max_length=5)
    reasoning: Literal['auto','low','medium','high'] = 'auto'

    @field_validator('strategy')
    @classmethod
    def valid_strategy(cls,value):
        if any(ord(c)<32 and c not in '\n\t' for c in value):
            raise ValueError('Control characters are not valid strategy text')
        return value

    @field_validator('exit_when')
    @classmethod
    def short_criteria(cls,value):
        if any(not s.strip() or len(s)>250 for s in value):
            raise ValueError('Exit conditions must be short and nonempty')
        return value


class ProfileRegistry:
    def __init__(self, directory=None, restore=False):
        self.directory, self.custom = directory, {}
        if restore and directory and (directory/'profiles.json').is_file():
            try:
                from .worktools import TOOLS
                for item in json.loads((directory/'profiles.json').read_text()):
                    if item.get('custom'):
                        self.install({k:v for k,v in item.items() if k!='custom'},TOOLS)
            except (ValueError,OSError): pass

    def install(self, spec, ceiling):
        from .minds import MODE_GUIDANCE
        from .worktools import TOOLS
        spec = ProfileSpec.model_validate(spec)
        if spec.name in MODE_GUIDANCE:
            raise ValueError('Custom profiles cannot replace built-in policy profiles')
        unknown = set(spec.tools)-set(TOOLS)
        if unknown:
            raise ValueError('Unknown tools: '+', '.join(sorted(unknown)))
        # Selecting a strategy cannot expand a node's runtime authority.
        spec = spec.model_copy(update={'tools':sorted(set(spec.tools)&set(ceiling))})
        self.custom[spec.name] = spec
        if self.directory:
            from .harness import _atomic
            _atomic(self.directory/'profiles.json',json.dumps(self.inventory(),ensure_ascii=False,indent=2))
        return spec.name

    def base(self, name):
        return self.custom[name].base if name in self.custom else name

    def tools(self, name):
        from .minds import MODE_TOOLS
        return frozenset(self.custom[name].tools) if name in self.custom else MODE_TOOLS[name]

    def prompt(self, name):
        from .minds import MODE_GUIDANCE
        if name not in self.custom:
            return MODE_GUIDANCE[name]
        profile = self.custom[name]
        return (MODE_GUIDANCE[profile.base]+'\nTASK STRATEGY (subordinate to operator constraints and runtime permissions):\n'+
                profile.strategy+'\nExit conditions: '+'; '.join(profile.exit_when))

    def contains(self, name):
        from .minds import MODE_GUIDANCE
        return name in MODE_GUIDANCE or name in self.custom

    def inventory(self):
        from .minds import MODE_GUIDANCE
        return ([{'name':name,'base':name,'custom':False} for name in MODE_GUIDANCE]+
                [dict(spec.model_dump(),custom=True) for spec in self.custom.values()])
