"""Small file-tool program executed inside Harbor containers, not on the host."""

def main():
    # This program runs inside the task container, with Python isolated from project imports.
    import base64,hashlib,json,os,tempfile
    from pathlib import Path
    payload=json.loads(base64.b64decode(__import__('sys').argv[1]))
    op,args=payload[:2]
    root=Path(payload[2] if len(payload)>2 else '/app').resolve()
    p=(root/str(args.get('path','.'))).resolve()
    if not p.is_relative_to(root): raise ValueError('File tools are scoped to /app')
    def safe(path):
        parts=path.relative_to(root).parts
        return not any(x in {'.git','.sea','__pycache__'} or x.lower().startswith('.env') or
            any(s in x.lower() for s in ['credential','secret','id_rsa','id_ed25519']) for x in parts)
    if not safe(p): raise ValueError('Protected path')
    result={'ok':True}
    if op=='exists': result['exists']=p.is_file()
    elif op in {'read','write','edit'}:
        if op=='write':
            if p.exists(): raise ValueError('File exists; read then edit it')
            text=str(args['content'])
        elif op=='edit':
            text=p.read_text();old=str(args['old'])
            if not old or text.count(old)!=1: raise ValueError('Old snippet must occur exactly once; inspect current source')
            text=text.replace(old,str(args['new']),1)
        else:
            if p.stat().st_size>2000000: raise ValueError('File too large; read through a targeted command')
            text=p.read_text()
        if op in {'write','edit'}:
            mode=p.stat().st_mode & 0o777 if p.exists() else 0o600
            p.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(dir=p.parent)
            with os.fdopen(fd,'w') as f: f.write(text)
            os.chmod(tmp,mode)
            os.replace(tmp,p)
        lines=text.splitlines();start=max(1,int(args.get('start',1)));end=min(int(args.get('end',start+119)),start+199)
        result.update(output=(f'FILE {p} lines {start}-{min(end,len(lines))} of {len(lines)}\n'+'\n'.join(lines[start-1:end]))
                             if op=='read' else 'Updated '+str(p),source_hash=hashlib.sha256(text.encode()).hexdigest())
    else:
        files=[]
        base=p if p.is_dir() else p.parent
        for current,dirs,names in os.walk(base):
            dirs[:]=[d for d in dirs if safe(Path(current)/d)]
            for n in names:
                q=Path(current)/n
                if q.is_file() and safe(q): files.append(q)
                if len(files)>=1000: break
            if len(files)>=1000: break
        if op=='tree': result['output']='\n'.join(str(q.relative_to(root)) for q in files[:250])
        elif op in {'search','grep'}:
            pattern=str(args.get('pattern',''));hits=[]
            for q in files:
                if q.stat().st_size>200000: continue
                try:
                    for i,line in enumerate(q.read_text().splitlines(),1):
                        if pattern.casefold() in line.casefold(): hits.append(f'{q}:{i}: {line[:250]}')
                        if len(hits)>=60: break
                except (UnicodeError,OSError): continue
                if len(hits)>=60: break
            result['output']='\n'.join(hits) or 'No matches'
        else: raise ValueError('Unknown file tool')
    print(json.dumps(result))

if __name__=="__main__": main()
