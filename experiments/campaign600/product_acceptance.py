"""External product acceptance. Run inside the product container, not on the host.

These tests were written independently of the generated tests and are not put in
the model's project during its initial attempts. They are finite checks, not an audit certificate.
"""
import importlib
import json
import re
import sys
import shutil
import stat
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setenv('DATABASE_PATH',str(tmp_path/'acceptance.sqlite3'))
    monkeypatch.setenv('APP_ENV','development')
    monkeypatch.setenv('SECRET_KEY','acceptance-only-random-shaped-key-0123456789abcdef')
    monkeypatch.setenv('ADMIN_USERNAME','owner')
    monkeypatch.setenv('ADMIN_PASSWORD','Acceptance-owner-password-9307!')
    monkeypatch.setenv('SEED_DEMO','0')
    for name in list(sys.modules):
        if name=='app' or name.startswith('app.'):monkeypatch.delitem(sys.modules,name)
    return importlib.import_module('app.main').app


@pytest.fixture
def client(app):
    with TestClient(app) as c:yield c


def token(c):
    r=c.get('/api/session');assert r.status_code==200,r.text
    return {'X-CSRF-Token':r.json()['csrf_token']}


def post(c,url,data=None,method='post'):
    return getattr(c,method)(url,headers=token(c),json=data or {})


def admin(c):
    r=post(c,'/api/auth/login',{'username':'owner','password':'Acceptance-owner-password-9307!'})
    assert r.status_code==200,r.text
    assert c.get('/api/session').json()['user']['is_admin'] is True


def register(c,name):
    r=post(c,'/api/auth/register',{'username':name,'password':'Reader-password-8462!'})
    assert r.status_code in (200,201),r.text


def feed(c,**params):
    r=c.get('/api/feed',params=params);assert r.status_code==200,r.text
    data=r.json();assert isinstance(data['items'],list) and 'next_cursor' in data
    return data


def add(c,title='Video alpha',topics=None,url='https://www.youtube.com/watch?v=dQw4w9WgXcQ'):
    r=post(c,'/api/admin/videos',{'title':title,'url':url,'description':'Description '+title,'topics':topics or ['Science']})
    assert r.status_code in (200,201),r.text
    rows=feed(c,q=title)['items'];found=[v for v in rows if v['title']==title]
    assert len(found)==1,rows
    return found[0]['id']


def items(response):
    data=response.json()
    if isinstance(data,list):return data
    for key in ('items','comments','reports','topics'):
        if key in data:return data[key]
    raise AssertionError(data)


def feedback(c,vid,kind,**kwargs):
    return post(c,f'/api/videos/{vid}/feedback',{'kind':kind,**kwargs})


def test_health_and_guest_session_are_real(client):
    assert client.get('/api/health').json()['status']=='ok'
    r=client.get('/api/session');j=r.json()
    assert j['user'] is None and len(j['csrf_token'])>=20
    assert client.get('/api/session').json()['csrf_token']==j['csrf_token']
    cookie=r.headers['set-cookie'].lower()
    assert 'httponly' in cookie and 'samesite=' in cookie
    assert feed(client)['items']==[]


def test_csrf_cannot_be_omitted_or_invented(client):
    token(client)
    for headers in ({},{'X-CSRF-Token':'invented'}):
        r=client.post('/api/auth/register',headers=headers,json={'username':'testuser','password':'Reader-password-8462!'})
        assert r.status_code==403,r.text
    register(client,'testuser')


def test_register_login_logout_and_session_rotation(client):
    token(client);before=dict(client.cookies)
    register(client,'alice')
    after=dict(client.cookies)
    assert before!=after,'Login/registration must rotate guest session identifier'
    assert client.get('/api/session').json()['user']['username']=='alice'
    assert post(client,'/api/auth/logout').status_code in (200,204)
    assert client.get('/api/session').json()['user'] is None
    wrong=post(client,'/api/auth/login',{'username':'alice','password':'wrong-password-value'})
    assert wrong.status_code in (400,401,403)
    assert post(client,'/api/auth/login',{'username':'alice','password':'Reader-password-8462!'}).status_code==200


def test_login_throttling_is_enforced(client):
    statuses=[]
    for _ in range(20):
        statuses.append(post(client,'/api/auth/login',{'username':'nobody','password':'wrong-password-value'}).status_code)
    assert 429 in statuses,statuses


def test_registration_does_not_accept_admin_role(client):
    r=post(client,'/api/auth/register',{'username':'attacker','password':'Reader-password-8462!','is_admin':True})
    assert r.status_code in (200,201,400,422)
    session=client.get('/api/session').json()
    if session['user']:assert session['user']['is_admin'] is False
    assert client.get('/api/admin/reports').status_code in (401,403)


def test_non_admin_cannot_mutate_catalog(client):
    register(client,'reader')
    data={'title':'forbidden','url':'https://example.com/a.mp4','topics':['Science']}
    assert post(client,'/api/admin/videos',data).status_code==403
    assert client.patch('/api/admin/videos/1',headers=token(client),json=data).status_code==403
    assert client.delete('/api/admin/videos/1',headers=token(client)).status_code==403


@pytest.mark.parametrize('url',[
    'javascript:alert(1)','data:text/html,<script>alert(1)</script>','file:///etc/passwd',
    'https://localhost/v.mp4','https://host.local/v.mp4','https://127.0.0.1/v.mp4',
    'https://127.1/v.mp4','https://2130706433/v.mp4','https://10.0.0.3/v.mp4',
    'https://0177.0.0.1/v.mp4','https://0x7f.0.0.1/v.mp4',
    'https://127.0.0.1./v.mp4','https://host.home.arpa/v.mp4',
    'https://169.254.169.254/v.mp4','https://[::1]/v.mp4',
    'https://user:password@example.com/v.mp4','http://example.com/v.mp4',
    '<iframe src="https://youtube.com"></iframe>',
])
def test_unsafe_external_links_are_rejected(client,url):
    admin(client)
    r=post(client,'/api/admin/videos',{'title':'Unsafe','url':url,'description':'','topics':['Science']})
    assert 400<=r.status_code<500,r.text
    assert feed(client)['items']==[]


@pytest.mark.parametrize('url',[
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ','https://youtu.be/dQw4w9WgXcQ',
    'https://www.youtube.com/shorts/dQw4w9WgXcQ','https://vimeo.com/76979871',
    'https://media.example.com/clip.mp4','https://media.example.com/clip.webm?version=2',
])
def test_supported_links_form_a_real_catalog(client,url):
    admin(client);vid=add(client,url=url)
    row=feed(client)['items'][0]
    assert row['id']==vid and row['provider'] and row['reason']
    assert {'title','url','topics','liked','saved'}<=row.keys()


def test_topics_search_and_stable_pagination(client):
    admin(client)
    expected=[]
    for i in range(7):expected.append(add(client,title=f'clip-{i:02}',topics=['Science' if i%2 else 'Music']))
    science=feed(client,topic='Science')['items']
    assert len(science)==3 and all('Science' in v['topics'] for v in science)
    assert [v['title'] for v in feed(client,q='clip-03')['items']]==['clip-03']
    seen=[];cursor=None
    for _ in range(10):
        query={'sort':'new','limit':2}
        if cursor:query['cursor']=cursor
        page=feed(client,**query);seen.extend(v['id'] for v in page['items']);cursor=page['next_cursor']
        if not cursor:break
    assert len(seen)==len(set(seen))==7 and set(seen)==set(expected)


def test_feedback_toggles_deduplicate_and_saved_persists(client):
    admin(client);vid=add(client)
    register_response=post(client,'/api/auth/logout');assert register_response.status_code in (200,204)
    register(client,'alice')
    for _ in range(3):assert feedback(client,vid,'like',value=True,event_id='like-1').status_code==200
    assert feed(client)['items'][0]['liked'] is True
    assert feedback(client,vid,'like',value=False,event_id='unlike-2').status_code==200
    assert feed(client)['items'][0]['liked'] is False
    assert feedback(client,vid,'save',value=True,event_id='save-3').status_code==200
    assert [v['id'] for v in items(client.get('/api/saved'))]==[vid]
    post(client,'/api/auth/logout')
    token(client)  # A fresh guest session is required by the documented API.
    assert items(client.get('/api/saved'))==[]
    post(client,'/api/auth/login',{'username':'alice','password':'Reader-password-8462!'})
    assert [v['id'] for v in items(client.get('/api/saved'))]==[vid]


def test_guest_feedback_migrates_and_does_not_leak_to_other_user(client,app):
    admin(client);vid=add(client);post(client,'/api/auth/logout')
    assert feedback(client,vid,'save',value=True,event_id='guest-save').status_code==200
    register(client,'alice')
    assert [v['id'] for v in items(client.get('/api/saved'))]==[vid]
    with TestClient(app) as other:
        register(other,'bob');assert items(other.get('/api/saved'))==[]
        assert feedback(other,vid,'like',value=True,event_id='guest-save').status_code==200
    assert feed(client)['items'][0]['liked'] is False


def test_hidden_video_and_privacy_reset(client):
    admin(client);vid=add(client);post(client,'/api/auth/logout');token(client)
    assert feedback(client,vid,'hide',value=True,event_id='hide-once').status_code==200
    assert feed(client)['items']==[]
    assert post(client,'/api/privacy/reset').status_code in (200,204)
    assert [v['id'] for v in feed(client)['items']]==[vid]


def test_watch_input_is_bounded(client):
    admin(client);vid=add(client)
    for value in (-1,1e30,'not-a-number'):
        r=feedback(client,vid,'watch',watch_seconds=value,event_id='watch-'+str(value))
        assert 400<=r.status_code<500,r.text
    assert feedback(client,vid,'watch',watch_seconds=12,event_id='valid-watch').status_code==200


def test_recommendations_react_to_topic_feedback(client):
    admin(client);science=add(client,'science-one',['Science']);music=add(client,'music-two',['Music'])
    post(client,'/api/auth/logout');token(client)
    assert feedback(client,science,'like',value=True,event_id='like-science').status_code==200
    row=feed(client,sort='for_you')['items'][0]
    assert row['id']==science and row['reason']
    assert feedback(client,science,'hide',value=True,event_id='hide-science').status_code==200
    assert [v['id'] for v in feed(client,sort='for_you')['items']]==[music]


def test_comment_ownership_and_admin_moderation(client,app):
    admin(client);vid=add(client);post(client,'/api/auth/logout')
    assert post(client,f'/api/videos/{vid}/comments',{'text':'guest'}).status_code in (401,403)
    register(client,'alice')
    r=post(client,f'/api/videos/{vid}/comments',{'text':'A useful comment'})
    assert r.status_code in (200,201),r.text
    comments=items(client.get(f'/api/videos/{vid}/comments'));assert len(comments)==1
    cid=comments[0]['id']
    with TestClient(app) as other:
        register(other,'bob')
        assert other.delete(f'/api/comments/{cid}',headers=token(other)).status_code in (403,404)
    assert client.delete(f'/api/comments/{cid}',headers=token(client)).status_code in (200,204)
    assert items(client.get(f'/api/videos/{vid}/comments'))==[]
    post(client,f'/api/videos/{vid}/comments',{'text':'Moderate this'})
    cid=items(client.get(f'/api/videos/{vid}/comments'))[0]['id']
    post(client,'/api/auth/logout');admin(client)
    assert client.delete(f'/api/comments/{cid}',headers=token(client)).status_code in (200,204)


def test_reports_are_visible_only_to_admin(client,app):
    admin(client);vid=add(client)
    with TestClient(app) as guest:
        token(guest);r=feedback(guest,vid,'report',reason='Broken link',event_id='report-once')
        assert r.status_code in (200,201),r.text
        assert guest.get('/api/admin/reports').status_code in (401,403)
    reports=items(client.get('/api/admin/reports'));assert len(reports)==1
    r=post(client,f"/api/admin/reports/{reports[0]['id']}",{'status':'resolved'},method='patch')
    assert r.status_code in (200,204),r.text


def test_admin_edit_and_delete_reach_feed(client):
    admin(client);vid=add(client)
    r=post(client,f'/api/admin/videos/{vid}',{'title':'Renamed','topics':['Math']},method='patch')
    assert r.status_code in (200,204),r.text
    assert feed(client,q='Renamed',topic='Math')['items'][0]['id']==vid
    assert client.delete(f'/api/admin/videos/{vid}',headers=token(client)).status_code in (200,204)
    assert feed(client)['items']==[]


def test_delete_account_preserves_other_users_and_catalog(client,app):
    admin(client);vid=add(client);post(client,'/api/auth/logout');register(client,'alice')
    feedback(client,vid,'save',value=True,event_id='save-delete')
    assert client.delete('/api/account',headers=token(client)).status_code in (200,204)
    assert client.get('/api/session').json()['user'] is None
    assert post(client,'/api/auth/login',{'username':'alice','password':'Reader-password-8462!'}).status_code in (400,401,403)
    with TestClient(app) as other:
        admin(other);assert [v['id'] for v in feed(other)['items']]==[vid]


def test_duplicate_event_race_does_not_turn_into_server_errors(client):
    admin(client);vid=add(client);headers=token(client)
    def send(_):return client.post(f'/api/videos/{vid}/feedback',headers=headers,
        json={'kind':'like','value':True,'event_id':'concurrent-one'}).status_code
    with ThreadPoolExecutor(max_workers=4) as executor:statuses=list(executor.map(send,range(8)))
    assert all(s in (200,409) for s in statuses),statuses
    assert feed(client)['items'][0]['liked'] is True


def test_frontend_and_local_assets_are_delivered(client):
    r=client.get('/');assert r.status_code==200
    assert 'text/html' in r.headers['content-type'] and len(r.text)>1000
    assert '<html' in r.text.lower() and 'viewport' in r.text
    assets=re.findall(r'(?:src|href)=["\'](/[^"\']+\.(?:css|js))(?:\?[^"\']*)?["\']',r.text)
    assert assets,'At least one real JS or CSS asset must be served'
    for asset in assets:
        result=client.get(asset);assert result.status_code==200 and len(result.content)>100


def test_invalid_json_and_large_requests_fail_cleanly(client):
    admin(client);headers=token(client);headers['Content-Type']='application/json'
    assert 400<=client.post('/api/admin/videos',headers=headers,content='{bad').status_code<500
    assert 400<=client.post('/api/admin/videos',headers=headers,content=json.dumps({'title':'x'*100000})).status_code<500
    assert client.get('/api/health').status_code==200


def test_cursor_is_bound_to_user_and_query(client,app):
    admin(client)
    for i in range(3):add(client,title=f'cursor-{i}')
    page=feed(client,limit=1);cursor=page['next_cursor'];assert cursor
    assert client.get('/api/feed',params={'cursor':cursor,'q':'different'}).status_code==422
    assert client.get('/api/feed',params={'cursor':cursor+'tamper'}).status_code==422
    with TestClient(app) as other:
        token(other)
        assert other.get('/api/feed',params={'cursor':cursor}).status_code==422


def test_admin_hide_and_restore_reach_feed(client):
    admin(client);vid=add(client)
    assert post(client,f'/api/admin/videos/{vid}',{'hidden':True},method='patch').status_code==200
    assert feed(client)['items']==[]
    assert post(client,f'/api/admin/videos/{vid}',{'hidden':False},method='patch').status_code==200
    assert feed(client)['items'][0]['id']==vid


def test_production_cookie_flags_and_cross_site_headers(tmp_path,monkeypatch):
    monkeypatch.setenv('DATABASE_PATH',str(tmp_path/'production.sqlite3'))
    monkeypatch.setenv('APP_ENV','production')
    monkeypatch.setenv('SECRET_KEY','production-test-secret-only-0123456789abcdef')
    monkeypatch.setenv('SEED_DEMO','0')
    for name in list(sys.modules):
        if name=='app' or name.startswith('app.'):monkeypatch.delitem(sys.modules,name)
    application=importlib.import_module('app.main').app
    with TestClient(application,base_url='https://flow.example.org') as c:
        r=c.get('/api/session')
        assert all(flag in r.headers['set-cookie'].lower() for flag in ('secure','httponly','samesite=lax'))
        assert 'no-store' in r.headers.get('cache-control','')
        assert c.get('/').headers.get('x-content-type-options')=='nosniff'


@pytest.mark.parametrize('authenticated',[False,True])
def test_personal_hide_removes_saved_video_without_affecting_other_saved(client,authenticated):
    admin(client);hidden=add(client,title='Hide this saved video');keep=add(client,title='Keep this saved video')
    post(client,'/api/auth/logout')
    if authenticated:register(client,'savedreader')
    for vid in (hidden,keep):
        assert feedback(client,vid,'save',value=True,event_id=f'save-hidden-{vid}').status_code==200
    assert {v['id'] for v in items(client.get('/api/saved'))}=={hidden,keep}
    assert feedback(client,hidden,'hide',value=True,event_id='hide-saved-once').status_code==200
    assert {v['id'] for v in items(client.get('/api/saved'))}=={keep}
    assert hidden not in {v['id'] for v in feed(client)['items']}


def test_configuration_generator_executes_safely_and_creates_separate_assignments(tmp_path):
    # Test copies execute only in container scratch; no production .env is created.
    configurations=[]
    for index in range(2):
        folder=tmp_path/str(index);folder.mkdir()
        script=folder/'setup.py';shutil.copyfile(Path('/work/setup.py'),script)
        imported=subprocess.run([sys.executable,'-I','-c',"import runpy; runpy.run_path('setup.py')"],
                                cwd=folder,capture_output=True,timeout=10)
        assert imported.returncode==0 and not (folder/'.env').exists()
        result=subprocess.run([sys.executable,'-I',str(script),'flow.example.org'],
                              cwd=folder,capture_output=True,timeout=10)
        assert result.returncode==0
        target=folder/'.env';raw=target.read_bytes()
        assert stat.S_IMODE(target.stat().st_mode)==0o600
        lines=raw.decode().splitlines()
        assert len(lines)==7,'Configuration must contain seven separate assignments'
        values=dict(line.split('=',1) for line in lines)
        assert set(values)=={'DOMAIN','APP_ENV','DATABASE_PATH','ADMIN_USERNAME','SECRET_KEY','ADMIN_PASSWORD','SEED_DEMO'}
        assert values['DOMAIN']=='flow.example.org' and values['APP_ENV']=='production'
        assert len(values['SECRET_KEY'])>=32 and len(values['ADMIN_PASSWORD'])>=24
        assert b'\\n' not in raw
        again=subprocess.run([sys.executable,'-I',str(script),'other.example.org'],
                             cwd=folder,capture_output=True,timeout=10)
        assert again.returncode!=0 and target.read_bytes()==raw
        configurations.append(values)
    assert configurations[0]['SECRET_KEY']!=configurations[1]['SECRET_KEY']
    assert configurations[0]['ADMIN_PASSWORD']!=configurations[1]['ADMIN_PASSWORD']
