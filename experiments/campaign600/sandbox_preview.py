"""Browser test bridge to real ASGI code in a networkless, read-only container.

No production image build, no host import of generated code, no product edits.
The HTTP listener binds only loopback. This is test infrastructure, not deployment.
"""
import argparse
import base64
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import threading
import uuid

CHILD='''
import base64,json,sys
sys.path.insert(0,'/work')
from app.main import app
from fastapi.testclient import TestClient
with TestClient(app,base_url='http://127.0.0.1:8808') as client:
 for line in sys.stdin:
  try:
   v=json.loads(line)
   client.cookies.clear()
   response=client.request(v['method'],v['path'],headers=v['headers'],content=base64.b64decode(v['body']),follow_redirects=False)
   print(json.dumps({'status':response.status_code,'headers':list(response.headers.multi_items()),'body':base64.b64encode(response.content).decode()}),flush=True)
  except Exception as exc:
   print(json.dumps({'status':500,'headers':[['Content-Type','application/json']],'body':base64.b64encode(json.dumps({'detail':type(exc).__name__+': '+str(exc)}).encode()).decode()}),flush=True)
'''


def main(args):
    project=args.project.resolve();assert (project/'app/main.py').is_file()
    name='sea-flow-browser-'+uuid.uuid4().hex[:10]
    argv=['docker','run','--rm','-i','--name',name,'--network','none','--read-only',
          '--cpus','1','--memory','512m','--pids-limit','64','--cap-drop','ALL',
          '--security-opt','no-new-privileges','--tmpfs','/tmp:rw,nosuid,size=64m',
          '--mount',f'type=bind,src={project},dst=/work,readonly',
          '-e','DATABASE_PATH=/tmp/preview.sqlite3','-e','APP_ENV=development',
          '-e','SECRET_KEY=preview-only-secret-0123456789abcdef',
          '-e','ADMIN_USERNAME=owner','-e','ADMIN_PASSWORD=Preview-only-password-8417!',
          '-e','SEED_DEMO=1','sea-campaign600-test:20261107','python','-u','-c',CHILD]
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def proxy(self):
            length=int(self.headers.get('Content-Length','0'))
            if length>65536:self.send_error(413);return
            payload={'method':self.command,'path':self.path,'headers':dict(self.headers),
                     'body':base64.b64encode(self.rfile.read(length)).decode()}
            with lock:
                process.stdin.write(json.dumps(payload)+'\n');process.stdin.flush()
                line=process.stdout.readline()
            if not line:self.send_error(502,'Sandbox stopped');return
            result=json.loads(line);body=base64.b64decode(result['body'])
            self.send_response(result['status'])
            for key,value in result['headers']:
                if key.lower() not in {'content-length','transfer-encoding','connection'}:self.send_header(key,value)
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            if self.command!='HEAD':self.wfile.write(body)
        do_GET=do_POST=do_PATCH=do_DELETE=do_HEAD=proxy
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    print(json.dumps({'preview':f'http://127.0.0.1:{args.port}','container':name,'scope':'test bridge; not production Uvicorn or Dockerfile validation'}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        process.stdin.close()
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('project',type=Path);p.add_argument('--port',type=int,default=8808)
    main(p.parse_args())
