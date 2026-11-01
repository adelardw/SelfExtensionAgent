"""Parse an explicit JSON key-list notation; leave arbitrary prose to the contract model."""
import re


def _json_deliveries(text):
    # This is a syntax parser, not a semantic classifier. Only fully recognized
    # instructions are authoritative. Never infer aliases or benchmark-specific names.
    pattern=r'\b(?:return|respond(?:\s+with)?|output|give|верни(?:те)?|выдай(?:те)?|возврати(?:те)?)\s+(?:a\s+)?JSON(?:\s+final\s+answer|\s+object)?\s+(?:with\s+(?:(?:keys|fields)\s+)?|с\s+(?:(?:ключами|полями)\s+)?)'
    matches=list(re.finditer(pattern,text,re.I))
    for match in reversed(matches):
        tail=text[match.end():];flat=[];depth=0;quote=None;end=len(text)
        for offset,ch in enumerate(tail):
            if quote:
                flat.append(ch)
                if ch==quote:quote=None
            elif ch in '\"`' and depth==0:quote=ch;flat.append(ch)
            elif ch=='(':depth+=1
            elif ch==')':
                if depth==0:end=match.end()+offset;break
                depth-=1
            elif depth:continue
            elif ch in '.;\n':end=match.end()+offset;break
            else:flat.append(ch)
        if depth or quote:continue
        chunks=re.split(r'\s*,\s*(?:(?:and|и)\s+)?|\s+(?:and|и)\s+', ''.join(flat).strip(),flags=re.I)
        keys=[]
        for chunk in chunks:
            chunk=chunk.strip()
            if not chunk:continue
            if len(chunk)>=2 and chunk[0]==chunk[-1] and chunk[0] in '\"`':chunk=chunk[1:-1]
            elif not re.fullmatch(r'[\w-]+',chunk):break
            if not 0<len(chunk)<=100:break
            keys.append(chunk)
        else:
            if keys and len(keys)==len(set(keys)) and len(keys)<=24:yield keys,match.start(),end,match.end()


def explicit_json_keys(text):
    for keys,_,_,_ in _json_deliveries(text):return keys
    return None


def json_delivery_only(text):
    """Recognize an entire delivery instruction, not a compound task or prose.

    This syntactic subset can establish answer shape, not authorization to create
    files/tests or integrate workers. Unrecognized phrasing remains semantic.
    """
    return any(not text[:start].strip() and not text[end:].strip(' .;\t\r\n')
               for _,start,end,_ in _json_deliveries(text))


FRACTION_PATTERN=r'^-?[0-9]+/[1-9][0-9]*$'


def _field_shape(description):
    text=' '.join(description.lower().split())
    if text in {'integer','int','целое','целое число'}:return {'type':'integer'}
    if text in {'string','text','строка'}:return {'type':'string'}
    if text in {'boolean','bool','логическое значение'}:return {'type':'boolean'}
    if text in {'number','float','число'}:return {'type':'number'}
    if text in {'object','mapping','dictionary','объект','словарь'}:return {'type':'object'}
    if text in {'p/q','fraction p/q','reduced fraction p/q','несократимая дробь p/q'}:
        return {'type':'string','pattern':FRACTION_PATTERN,'description':
            'Reduced fraction p/q' if text in {'reduced fraction p/q','несократимая дробь p/q'} else 'Fraction p/q'}
    if text in {'list of integers','array of integers','список целых чисел'}:
        return {'type':'array','items':{'type':'integer'}}
    if text in {'list','array','список'} or text.startswith(('list of ','array of ','список ')):
        return {'type':'array'}
    return {}


def explicit_json_fields(text):
    """Compile only recognized parenthesized field types, independent of key names."""
    for keys,_,end,body in _json_deliveries(text):
        fields={}
        for key in keys:
            name=re.escape(key)
            match=re.search(r'(?:^|,|\band\b|\bи\b)\s*(?:'+name+r'|"'+name+r'"|`'+name+r'`)\s*\(([^()]*)\)',
                            text[body:end],re.I)
            fields[key]=_field_shape(match.group(1)) if match else {}
        return fields
    return None


def delivery_fields(task,instructions,keys):
    # Unparsed later steering may change a field's type without changing its key.
    # Do not impose a stale type from an earlier instruction in that situation.
    parsed=explicit_json_fields(instructions[-1]['text'] if instructions else task)
    return parsed if parsed is not None and set(parsed)==set(keys) else {}


def validate_delivery_fields(value,fields):
    import math
    def check(actual,shape,path):
        kind=shape.get('type')
        valid=(kind is None or kind=='string' and isinstance(actual,str)
            or kind=='integer' and type(actual) is int
            or kind=='number' and type(actual) in (int,float) and (type(actual) is int or math.isfinite(actual))
            or kind=='boolean' and type(actual) is bool or kind=='array' and isinstance(actual,list)
            or kind=='object' and isinstance(actual,dict))
        if not valid:
            raise ValueError(f'Field {path} requires {shape.get("description",kind)}; received {str(actual)[:100]}. '
                             'Preserve the computed value; fix its requested representation.')
        if shape.get('pattern') and not re.fullmatch(shape['pattern'],actual):
            raise ValueError(f'Field {path} requires a fraction string p/q with a positive denominator.')
        if shape.get('description')=='Reduced fraction p/q':
            numerator,denominator=map(int,actual.split('/'))
            if math.gcd(numerator,denominator)!=1:raise ValueError(f'Field {path} requires a reduced fraction p/q.')
        if kind=='array' and shape.get('items'):
            for i,item in enumerate(actual):check(item,shape['items'],f'{path}[{i}]')
        if kind=='object':
            if len(actual)<shape.get('minProperties',0) or len(actual)>shape.get('maxProperties',len(actual)):
                raise ValueError(f'Field {path} has an invalid number of properties')
            missing=set(shape.get('required',[]))-set(actual)
            if missing:raise ValueError(f'Field {path} is missing required properties: '+', '.join(sorted(missing)))
            properties=shape.get('properties',{})
            additional=shape.get('additionalProperties',True)
            for name,item in actual.items():
                if name in properties:check(item,properties[name],f'{path}.{name}')
                elif additional is False:raise ValueError(f'Field {path} has an unexpected property: {name}')
                elif isinstance(additional,dict):check(item,additional,f'{path}.{name}')
    if fields and not isinstance(value,dict):raise ValueError('Final JSON answer must be an object')
    for key,shape in fields.items():
        if key in value:check(value[key],shape,key)
