"""Replay one recorded critic request to verify transport; never touches product."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import jsonschema
from pydantic import BaseModel,ConfigDict,model_validator

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from codex_behavior_eval import CodexInference


async def main():
    base=ROOT/'data/eval/campaign600-20261107'
    request=json.loads((base/'product-attempt-3/calls/call-044/request.json').read_text())
    wire=request['schema']
    class RecordedReview(BaseModel):
        model_config=ConfigDict(extra='allow')
        @classmethod
        def model_json_schema(cls,**kwargs):return copy.deepcopy(wire)
        @model_validator(mode='before')
        @classmethod
        def validate_recorded(cls,value):
            jsonschema.validate(value,wire)
            return value
    shared={'calls':0,'tokens':0}
    output=base/'critic-transport-replay'
    output.mkdir(exist_ok=False)
    model=CodexInference(output/'calls',shared,max_calls=1,shared_call_limit=1,
                         shared_token_limit=180000,timeout_seconds=180,output_bound=12000,payload_object=True)
    result=await model(request['messages'],RecordedReview,'medium')
    record={'response':result.model_dump(),'usage':shared,
            'scope':'Historical critic request replay; not a current completion review or product pass'}
    (output/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'transport':'accepted','ready':result.model_dump()['ready'],'usage':shared}))


if __name__=='__main__':asyncio.run(main())
