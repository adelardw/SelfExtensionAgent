"""Independent browser scenarios; backend runs through sandbox_preview.py.

External media requests are stubbed. This validates application interaction, not
the availability or playback of third-party videos, public TLS or deployment.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

BASE='http://127.0.0.1:8808'


def check_card_geometry(page):
    measurements=page.locator('.video-card').first.evaluate('''card => {
      const f=document.querySelector('#feed').getBoundingClientRect();
      const a=card.querySelector('.actions').getBoundingClientRect();
      const p=card.querySelector('.player').getBoundingClientRect();
      return {width:innerWidth,height:innerHeight,feed_top:f.top,feed_bottom:f.bottom,
        actions_top:a.top,actions_bottom:a.bottom,player_top:p.top,player_bottom:p.bottom,
        controls_fully_visible:a.top>=f.top-1 && a.bottom<=Math.min(f.bottom,innerHeight)+1,
        player_visible:p.height>50 && p.top>=f.top-1 && p.bottom<=Math.min(f.bottom,innerHeight)+1};
    }''')
    assert measurements['controls_fully_visible'],measurements
    assert measurements['player_visible'],measurements
    return measurements


def check_real_media_clock(browser, route):
    """Use actual Chromium media time from a local stream; no external video."""
    context=browser.new_context(viewport={'width':1440,'height':1000})
    context.route('**/*',route)
    page=context.new_page();watches=[];errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    def capture(request):
        if request.method=='POST' and request.url.endswith('/feedback'):
            body=request.post_data_json
            if body.get('kind')=='watch':watches.append(body)
    page.on('request',capture)
    try:
        page.goto(BASE);expect(page.locator('.video-card')).to_have_count(3)
        page.evaluate("state.items.find(e=>e.item.provider==='direct').card.scrollIntoView({behavior:'instant'})")
        expect(page.locator('video')).to_have_count(1)
        page.wait_for_timeout(1400)
        page.locator('#saved-nav').click()
        page.wait_for_timeout(100)
        assert not watches,('Blocked autoplay counted as watch',watches)
        page.locator('#feed-nav').click();expect(page.locator('.video-card')).to_have_count(3)
        page.evaluate("state.items.find(e=>e.item.provider==='direct').card.scrollIntoView({behavior:'instant'})")
        expect(page.locator('video')).to_have_count(1)
        page.evaluate('''async () => {
            const video=document.querySelector('video');
            const canvas=document.createElement('canvas');canvas.width=160;canvas.height=90;
            const ctx=canvas.getContext('2d');let tick=0;
            window.acceptanceTimer=setInterval(()=>{
                ctx.fillStyle=`hsl(${tick++%360} 60% 40%)`;ctx.fillRect(0,0,160,90);
            },50);
            window.acceptanceStream=canvas.captureStream(20);
            video.srcObject=window.acceptanceStream;video.muted=true;await video.play();
        }''')
        page.wait_for_function("document.querySelector('video').currentTime > 1.4")
        played=page.evaluate("() => {const v=document.querySelector('video');v.pause();return v.currentTime;}")
        page.wait_for_timeout(2200)
        page.locator('#saved-nav').click()
        page.wait_for_timeout(150)
        assert len(watches)==1,watches
        observed=watches[0]['watch_seconds']
        assert 0.7 <= observed <= played+0.5,(played,observed)
        assert not errors,errors
        return {'actual_media_seconds':played,'reported_watch_seconds':observed,
                'paused_interval_seconds':2.2,'blocked_autoplay_reported_watch':False,
                'source':'local canvas captureStream, real Chromium playback clock'}
    finally:
        page.evaluate('''() => {
            clearInterval(window.acceptanceTimer);
            window.acceptanceStream?.getTracks().forEach(track=>track.stop());
        }''')
        context.close()


def main(ROOT):
    ROOT.mkdir(parents=True,exist_ok=True)
    steps=[];errors=[];geometry=[];playback=None
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000})
        def route(req):
            if req.request.url.startswith(BASE):return req.continue_()
            if req.request.resource_type=='document':
                return req.fulfill(content_type='text/html',body='<html lang="ru"><body style="background:#172025;color:#acc7c0;display:grid;place-items:center;height:95vh;font:16px system-ui;text-align:center">Внешний плеер<br>Изолированная проверка интерфейса</body></html>')
            return req.abort()
        context.route('**/*',route)
        page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('dialog',lambda dialog:dialog.accept())
        page.set_default_timeout(15000)
        try:
            page.goto(BASE);expect(page.locator('.video-card')).to_have_count(3)
            expect(page.locator('#identity')).to_have_text('Гостевой просмотр')
            steps.append('guest_session_and_seeded_feed')
            first=page.locator('.video-card').first
            first.get_by_role('button',name='Нравится',exact=False).click()
            expect(first.get_by_role('button',name='Нравится',exact=False)).to_have_attribute('aria-pressed','true')
            first.get_by_role('button',name='Сохранить',exact=False).click()
            expect(first.get_by_role('button',name='Сохранено',exact=False)).to_have_attribute('aria-pressed','true')
            page.locator('#saved-nav').click();expect(page.locator('.video-card')).to_have_count(1)
            steps.append('guest_like_and_saved_feed')
            page.locator('#feed-nav').click();expect(page.locator('.video-card')).to_have_count(3)
            page.locator('#topics').get_by_role('button',name='Знакомство',exact=True).click()
            expect(page.locator('.video-card')).to_have_count(1)
            page.locator('#search').fill('this-title-does-not-exist')
            page.locator('#search-form').get_by_role('button').click()
            expect(page.locator('.video-card')).to_have_count(0)
            expect(page.locator('#feed-state')).to_contain_text('пока нет видео')
            page.locator('#search').fill('');page.locator('#search-form').get_by_role('button').click()
            page.locator('#topics').get_by_role('button',name='Все темы',exact=True).click()
            expect(page.locator('.video-card')).to_have_count(3)
            steps.append('topics_search_and_empty_state')
            page.locator('#auth-open').click();page.locator('#auth-mode').select_option('register')
            page.locator('#username').fill('browser-reader');page.locator('#password').fill('Browser-only-password-1846!')
            page.locator('#auth-form').get_by_role('button',name='Продолжить').click()
            expect(page.locator('#auth-dialog')).not_to_be_visible()
            expect(page.locator('#identity')).to_have_text('browser-reader')
            page.locator('#saved-nav').click();expect(page.locator('.video-card')).to_have_count(1)
            steps.append('registration_and_guest_preference_migration')
            page.locator('.video-card').first.get_by_role('button',name='Комментарии',exact=True).click()
            message='Useful <script>window.flowInjected=true</script> comment'
            page.locator('#comment-text').fill(message)
            page.locator('#comment-form').get_by_role('button',name='Опубликовать').click()
            expect(page.locator('#comments-list')).to_contain_text(message)
            assert page.evaluate('window.flowInjected === undefined')
            page.locator('#comments-list').get_by_role('button',name='Удалить комментарий').click()
            expect(page.locator('#comments-list')).not_to_contain_text(message)
            page.locator('#comments-dialog').get_by_role('button',name='Закрыть').click()
            steps.append('comments_persist_escape_text_and_delete')
            page.locator('#feed-nav').click();expect(page.locator('.video-card')).to_have_count(3)
            page.screenshot(path=str(ROOT/'desktop.png'),full_page=True)
            geometry.append(check_card_geometry(page))
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert page.locator('.player iframe,.player video').count()<=1
            page.locator('#next').click();page.wait_for_timeout(500)
            assert page.locator('.player iframe,.player video').count()<=1
            steps.append('desktop_layout_and_single_active_player')
            page.locator('#logout').click();expect(page.locator('#identity')).to_have_text('Гостевой просмотр')
            page.locator('#auth-open').click();page.locator('#auth-mode').select_option('login')
            page.locator('#username').fill('owner');page.locator('#password').fill('Preview-only-password-8417!')
            page.locator('#auth-form').get_by_role('button',name='Продолжить').click()
            expect(page.locator('#identity')).to_have_text('owner')
            page.locator('#admin-nav').click();expect(page.locator('#admin-view')).to_be_visible()
            page.locator('#video-title').fill('Browser acceptance video')
            page.locator('#video-url').fill('https://www.youtube.com/watch?v=M7lc1UVf-VE')
            page.locator('#video-topics').fill('Browser test')
            page.locator('#video-form').get_by_role('button',name='Сохранить видео').click()
            row=page.locator('#admin-catalog article').filter(has_text='Browser acceptance video')
            expect(row).to_have_count(1)
            row.get_by_role('button',name='Скрыть',exact=True).click()
            expect(row).to_contain_text('Скрыто администратором')
            row.get_by_role('button',name='Вернуть в ленту',exact=True).click()
            expect(row).to_contain_text('Опубликовано')
            row.get_by_role('button',name='Удалить видео',exact=True).click();expect(row).to_have_count(0)
            steps.append('admin_create_hide_restore_delete')
            mobile=browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
            mobile.route('**/*',route);m=mobile.new_page();m.on('pageerror',lambda error:errors.append(str(error)))
            m.goto(BASE);expect(m.locator('.video-card')).to_have_count(3)
            assert m.evaluate('document.documentElement.scrollWidth <= innerWidth')
            m.screenshot(path=str(ROOT/'mobile.png'),full_page=True)
            geometry.append(check_card_geometry(m))
            m.locator('.card-metadata').first.evaluate("el => { const p=document.createElement('p'); p.textContent='Длинное описание '.repeat(300); el.append(p); }")
            geometry.append(dict(check_card_geometry(m),scenario='long_description'))
            m.locator('#next').click();m.wait_for_timeout(500)
            assert m.locator('.player iframe,.player video').count()<=1
            mobile.close();steps.append('mobile_layout_and_navigation')
            playback=check_real_media_clock(browser,route)
            steps.append('real_local_playback_excludes_blocked_autoplay_and_pause')
            assert not errors,errors
            result={'passed':True,'steps':steps,'page_errors':errors}
        except Exception as exc:
            page.screenshot(path=str(ROOT/'failure.png'),full_page=True)
            result={'passed':False,'steps':steps,'page_errors':errors,'failure':type(exc).__name__+': '+str(exc)}
        finally:browser.close()
    result['scope']='Real harness app via networkless ASGI sandbox bridge; external media stubbed; no production Docker build/TLS verification'
    result['geometry']=geometry
    result['playback']=playback
    (ROOT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/eval/campaign600-20261107/browser'))
    main(parser.parse_args().output)
