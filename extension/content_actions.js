// Исполняется ВНУТРИ страницы (chrome.scripting.executeScript) — структурные действия:
// снапшот интерактивных элементов с номерами, клик/ввод/медиа/чтение по номеру.
// Зеркало _SNAPSHOT_JS/_MEDIA_JS/_READ_JS из src/browser_session.py.
globalThis.agentExec = function agentExec(req) {
  const A = "data-agent-i";

  // НАДЁЖНОЕ определение «играет ли звук», кросс-сайтово (Я.Музыка не держит обычный
  // <audio> с paused=false): 1) медиа-элементы; 2) navigator.mediaSession.playbackState
  // (ставят YM/YouTube/SoundCloud для системных медиа-контролов); 3) видимая кнопка «Пауза».
  function isPlaying() {
    if ([...document.querySelectorAll('audio,video')].some(m => !m.paused && !m.ended && m.currentTime > 0))
      return true;
    try { if (navigator.mediaSession && navigator.mediaSession.playbackState === 'playing') return true; }
    catch (e) {}
    const pause = /(pause|пауза|приостанов)/i;
    for (const el of document.querySelectorAll('button,[role="button"],[aria-label],[title]')) {
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) continue;
      const lbl = (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '').trim();
      if (pause.test(lbl)) return true;  // кнопка «Пауза» видна → значит что-то играет
    }
    return false;
  }
  function nowTitle() {
    try { const m = navigator.mediaSession && navigator.mediaSession.metadata;
      if (m && m.title) return m.title + (m.artist ? ' — ' + m.artist : ''); } catch (e) {}
    return document.title;
  }

  function snapshot(note) {
    const sel = 'a, button, input, textarea, select, [role="button"], [role="link"],' +
      '[role="tab"], [role="menuitem"], [role="checkbox"], [onclick],' +
      '[contenteditable="true"], audio, video, [role="searchbox"]';
    // Видимость: элемент имеет размеры и НЕ скрыт стилями. БЕЗ требования «в текущем
    // вьюпорте» — кнопки ниже фолда (напр. «В корзину» внизу модалки) тоже нужны: клик
    // по элементу через DOM работает и без прокрутки. Скрытые (display:none/0-размер) — мимо.
    const vis = el => { const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return false;
      const s = getComputedStyle(el);
      return s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'; };
    const txt = el => (el.getAttribute('aria-label') || el.innerText || el.value ||
      el.getAttribute('placeholder') || el.getAttribute('title') || '')
      .trim().replace(/\s+/g, ' ').slice(0, 80);
    const items = [];
    let i = 0;
    for (const el of document.querySelectorAll(sel)) {
      if (!vis(el)) continue;
      el.setAttribute(A, String(i));
      const tag = el.tagName.toLowerCase();
      let kind = tag === 'input' ? `input:${el.type || 'text'}` : tag;
      if (tag === 'audio' || tag === 'video') kind += el.paused ? ' (на паузе)' : ' (СЕЙЧАС ИГРАЕТ)';
      items.push(`  [${i}] ${kind}: ${txt(el) || '(без текста)'}`);
      if (++i >= 80) break;  // выше лимит: длинные меню/модалки с товарами и кнопкой внизу
    }
    const head = [`Страница: ${JSON.stringify(document.title)} · ${location.href}`];
    if (note) head.unshift(note);
    if (isPlaying()) head.push(`  ♪ ЗВУК ИГРАЕТ (${nowTitle()}) — цель достигнута, не перепроверяй`);
    if (!items.length) head.push('  (интерактивных элементов не видно — повтори see)');
    return head.concat(items).concat(
      ['Дальше: click(i) · type(i, текст) · media(pause|play) · scroll · see']).join('\n');
  }

  const el = req.item != null ? document.querySelector(`[${A}="${req.item}"]`) : null;
  try {
    if (req.action === 'see') return snapshot('');
    if (req.action === 'click') {
      if (!el) return 'Элемент не найден — сделай see заново.';
      el.click(); return snapshot(`Кликнул [${req.item}].`);
    }
    if (req.action === 'type') {
      if (!el) return 'Поле не найдено — сделай see заново.';
      const s = ((el.type||'')+' '+(el.name||'')+' '+(el.id||'')+' '+(el.autocomplete||'')).toLowerCase();
      if (el.type === 'password' || /cc-|card|cvc|cvv|карт|паспорт|passport|снилс|инн\b/.test(s))
        return '[ГРАНИЦА ПЕРСОНАЛЬНЫХ ДАННЫХ] поле пароля/карты/документа — агент не печатает. Попроси пользователя ввести самому.';
      el.focus(); el.value = req.text;
      el.dispatchEvent(new Event('input', {bubbles: true}));
      el.dispatchEvent(new Event('change', {bubbles: true}));
      if (req.submit) el.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', keyCode: 13, bubbles: true}));
      return snapshot(`Ввёл «${req.text}»` + (req.submit ? ' и Enter.' : '.'));
    }
    if (req.action === 'press') {
      document.activeElement?.dispatchEvent(new KeyboardEvent('keydown', {key: req.key, bubbles: true}));
      return snapshot(`Нажал ${req.key}.`);
    }
    if (req.action === 'scroll') {
      window.scrollBy(0, req.direction === 'up' ? -700 : 700);
      return snapshot(`Проскроллил ${req.direction}.`);
    }
    if (req.action === 'media') {
      const ms = [...document.querySelectorAll('audio, video')];
      let n = 0;
      for (const m of ms) {
        if (req.mediaAction === 'pause' && !m.paused) { m.pause(); n++; }
        else if (req.mediaAction === 'play' && m.paused) { m.play().catch(()=>{}); n++; }
        else if (req.mediaAction === 'toggle') { m.paused ? m.play().catch(()=>{}) : m.pause(); n++; }
        else if (req.mediaAction === 'mute') { m.muted = true; n++; }
        else if (req.mediaAction === 'unmute') { m.muted = false; n++; }
      }
      // PLAY без играющего звука → жмём кнопку воспроизведения (общая эвристика для
      // кастомных плееров: Я.Музыка/YouTube/SoundCloud — кнопка с aria-label/текстом play).
      if ((req.mediaAction === 'play' || req.mediaAction === 'toggle') && !isPlaying()) {
        const re = /(^|\b)(play|слуша|воспроизв|▶)/i;
        const cands = [...document.querySelectorAll(
          'button,[role="button"],a,[aria-label],[title],[data-test-id*="play" i]')];
        for (const el of cands) {
          const r = el.getBoundingClientRect();
          if (r.width < 1 || r.height < 1) continue;
          const label = (el.getAttribute('aria-label')||el.getAttribute('title')||
                         el.textContent||el.getAttribute('data-test-id')||'').trim();
          if (re.test(label)) { el.click(); n++; break; }
        }
      }
      // Медиа стартует не мгновенно — подождём ~1.5с перед вердиктом (по isPlaying).
      return new Promise(resolve => setTimeout(() => resolve(
        `${req.mediaAction}: действий ${n}; ` +
        (isPlaying() ? `♪ ЗВУК ИГРАЕТ (${nowTitle()})` : 'звук не пошёл — кликни конкретную кнопку плей из снапшота')
      ), 1500));
    }
    if (req.action === 'read') {
      // JS-тяжёлые сайты дорисовывают контент ПОСЛЕ load — ждём ~1.3с, затем читаем
      // (до 9000 симв.). Общий приём для любого источника, не под конкретный сайт.
      return new Promise(resolve => setTimeout(() => {
        const cl = document.body.cloneNode(true);
        cl.querySelectorAll('script,style,noscript,svg').forEach(e => e.remove());
        resolve(`Содержимое ${JSON.stringify(document.title)}:\n` +
          (cl.innerText || '').replace(/\n{3,}/g, '\n\n').trim().slice(0, 9000));
      }, 1300));
    }
    return 'Неизвестное действие.';
  } catch (e) { return 'Ошибка в странице: ' + e.message; }
};
