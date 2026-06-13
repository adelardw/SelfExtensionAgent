// Исполняется ВНУТРИ страницы (chrome.scripting.executeScript) — структурные действия:
// снапшот интерактивных элементов с номерами, клик/ввод/медиа/чтение по номеру.
// Зеркало _SNAPSHOT_JS/_MEDIA_JS/_READ_JS из src/browser_session.py.
globalThis.agentExec = function agentExec(req) {
  const A = "data-agent-i";

  // НАДЁЖНОЕ определение «играет ли звук», кросс-сайтово (Я.Музыка не держит обычный
  // <audio> с paused=false): 1) медиа-элементы; 2) navigator.mediaSession.playbackState
  // ИЛИ выставленные metadata (ставят YM/YouTube/Spotify для системных медиа-контролов).
  // Кнопка «Пауза» сама по себе НЕ доказательство: Spotify оптимистично переключает UI
  // без реального звука (живой false positive «ЗВУК ИГРАЕТ» при тишине) — честность важнее.
  function isPlaying() {
    if ([...document.querySelectorAll('audio,video')].some(m => !m.paused && !m.ended && m.currentTime > 0))
      return true;
    try {
      if (navigator.mediaSession && navigator.mediaSession.playbackState === 'playing') return true;
      // metadata выставлены И есть видимая кнопка «Пауза» — играет кастомный плеер
      // (YM: playbackState бывает 'none', но metadata живые при звуке).
      if (navigator.mediaSession && navigator.mediaSession.metadata &&
          navigator.mediaSession.metadata.title) {
        const pause = /(pause|пауза|приостанов)/i;
        for (const el of document.querySelectorAll('button,[role="button"],[aria-label],[title]')) {
          const r = el.getBoundingClientRect();
          if (r.width < 1 || r.height < 1) continue;
          const lbl = (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '').trim();
          if (pause.test(lbl)) return true;
        }
      }
    } catch (e) {}
    return false;
  }
  // [ГРАНИЦА ДЕНЕГ] кнопка НЕОБРАТИМОГО оформления/оплаты заказа — агент не нажимает САМ
  // (даже в auto: сохранённая карта = one-click заказ без спроса). Зеркало отказа type в картах.
  const _PAY_RE = new RegExp('оплатить|оформить заказ|подтвердить заказ|оформить и оплатить|'
    + 'оплата заказа|купить сейчас|заказать за\\s*\\d|оплатить\\s*\\d|place order|pay now|'
    + 'checkout|buy now', 'i');
  function payLabel(el) {
    if (!el) return '';
    const lbl = (el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('title')) ||
                 el.value || el.innerText || el.textContent || '').trim();
    return _PAY_RE.test(lbl) ? lbl.slice(0, 50) : '';
  }
  function payRefusal(lbl) {
    return '[ГРАНИЦА ОПЛАТЫ] это кнопка ОФОРМЛЕНИЯ/ОПЛАТЫ заказа («' + lbl + '») — агент НЕ '
      + 'оформляет платный заказ сам. Корзина готова: попроси пользователя подтвердить и нажать '
      + 'эту кнопку самому (ask_user), не нажимай за него.';
  }
  function nowTitle() {
    try { const m = navigator.mediaSession && navigator.mediaSession.metadata;
      if (m && m.title) return m.title + (m.artist ? ' — ' + m.artist : ''); } catch (e) {}
    return document.title;
  }

  // ── Извлечение элементов в стиле browser-use (надёжно, не самоделка-снапшот) ──
  const ITAGS = new Set(['a','button','input','select','textarea','summary','label','option','audio','video']);
  const IROLE = /^(button|link|tab|menuitem(checkbox|radio)?|checkbox|radio|switch|option|searchbox|textbox|combobox|slider|menuitemradio)$/i;

  function isInteractive(el) {
    const tag = el.tagName.toLowerCase();
    if (ITAGS.has(tag)) return true;
    const role = el.getAttribute('role');
    if (role && IROLE.test(role)) return true;
    if (el.hasAttribute('onclick') || el.getAttribute('contenteditable') === 'true') return true;
    const ti = el.getAttribute('tabindex');
    if (ti !== null && ti !== '-1') return true;
    try { if (getComputedStyle(el).cursor === 'pointer') return true; } catch (e) {}  // стилизованные div-кнопки
    return false;
  }
  function visible(el) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    let s; try { s = getComputedStyle(el); } catch (e) { return false; }
    return s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0';
  }
  function topmost(el, doc) {
    // Не перекрыт ли оверлеем (куки-баннер и т.п.). Под/над фолдом проверить нельзя —
    // включаем (клик по DOM работает и без прокрутки).
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (cy < 0 || cy > innerHeight || cx < 0 || cx > innerWidth) return true;
    let top; try { top = (doc || document).elementFromPoint(cx, cy); } catch (e) { return true; }
    if (!top) return true;
    return el === top || el.contains(top) || top.contains(el);
  }
  function elemText(el) {
    return (el.getAttribute('aria-label') || el.value || el.innerText ||
      el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || '')
      .trim().replace(/\s+/g, ' ').slice(0, 90);
  }
  // Обход дерева С ПРОНИКНОВЕНИЕМ в shadow DOM и same-origin iframe.
  // pierced=true → мы внутри shadow/iframe (там топмост-проверка по elementFromPoint
  // некорректна — координаты/пирсинг другие, поэтому её пропускаем).
  function walk(root, doc, pierced, visit) {
    let nodes; try { nodes = root.querySelectorAll('*'); } catch (e) { return; }
    for (const el of nodes) {
      visit(el, doc, pierced);
      if (el.shadowRoot) walk(el.shadowRoot, doc, true, visit);
      if (el.tagName === 'IFRAME') {
        let idoc; try { idoc = el.contentDocument; } catch (e) { idoc = null; }
        if (idoc && idoc.body) walk(idoc.body, idoc, true, visit);
      }
    }
  }

  function collect() {
    const cands = [];
    walk(document.documentElement, document, false, (el, doc, pierced) => {
      if (cands.length >= 250) return;
      if (isInteractive(el) && visible(el) && (pierced || topmost(el, doc))) cands.push(el);
    });
    // Берём ИННЕРМОСТ кликабельные (выкидываем обёртки-контейнеры).
    let leaves = cands.filter(el => !cands.some(o => o !== el && el.contains(o)));
    // ТОКЕН-ЭКОНОМИЯ: безымянные элементы без текста/лейбла малоценны (раздували контекст
    // до 151k и баны бюджета) — оставляем только осмысленные (с текстом) + поля/медиа.
    const meaningful = leaves.filter(el => {
      const t = el.tagName.toLowerCase();
      return elemText(el) || ['input', 'select', 'textarea', 'audio', 'video'].includes(t);
    });
    return (meaningful.length ? meaningful : leaves).slice(0, 120);
  }
  // Глубокий поиск ранее помеченного элемента (click/type после see) — через тот же обход.
  function findIndexed(i) {
    let found = null;
    walk(document.documentElement, document, false, (el) => {
      if (!found && el.getAttribute && el.getAttribute(A) === String(i)) found = el;
    });
    return found;
  }

  function snapshot(note) {
    walk(document.documentElement, document, false, (e) => {  // глубокая чистка прошлой разметки
      if (e.removeAttribute) e.removeAttribute(A);
    });
    const els = collect();
    const items = [];
    els.forEach((el, i) => {
      el.setAttribute(A, String(i));
      const tag = el.tagName.toLowerCase();
      let kind = tag === 'input' ? `input:${el.type || 'text'}` : (el.getAttribute('role') || tag);
      if (tag === 'audio' || tag === 'video') kind += el.paused ? ' (на паузе)' : ' (ИГРАЕТ)';
      items.push(`  [${i}] ${kind}: ${elemText(el) || '(без текста)'}`);
    });
    const head = [`Страница: ${JSON.stringify(document.title)} · ${location.href}`];
    if (note) head.unshift(note);
    if (isPlaying()) head.push(`  ♪ ЗВУК ИГРАЕТ (${nowTitle()}) — цель достигнута, не перепроверяй`);
    if (!items.length) head.push('  (интерактивных элементов не видно — повтори see)');
    return head.concat(items).concat(
      ['Дальше: click(i) · type(i, текст) · media(pause|play) · scroll · see']).join('\n');
  }

  const el = req.item != null ? findIndexed(req.item) : null;
  try {
    if (req.action === 'see') {
      // SPA (SoundCloud и пр.) гидрируются ПОСЛЕ load: пустой/почти пустой снапшот →
      // подождать и снять ещё раз (до 3 попыток). Универсально, не под конкретный сайт.
      const tryOnce = (left, resolve) => {
        const snap = snapshot(req.note || '');
        const items = (snap.match(/\n  \[/g) || []).length;
        if (items >= 3 || left <= 0) return resolve(snap);
        setTimeout(() => tryOnce(left - 1, resolve), 1200);
      };
      return new Promise(resolve => tryOnce(3, resolve));
    }
    if (req.action === 'click') {
      if (!el) return 'Элемент не найден — сделай see заново.';
      const pl = payLabel(el);
      if (pl) return payRefusal(pl);   // [ГРАНИЦА ОПЛАТЫ] финальный платный тап — за человеком
      try { el.scrollIntoView({ block: 'center', behavior: 'instant' }); } catch (e) {}
      el.click();
      return snapshot(`Кликнул [${req.item}].`);
    }
    if (req.action === 'clicksel') {
      // Детерминированный клик по первому видимому элементу под CSS-селектор (без LLM).
      // ЖДЁМ появления (SPA вроде YouTube дорисовывают результаты асинхронно) — до ~5с.
      const pickAndClick = () => {
        for (const c of document.querySelectorAll(req.selector)) {
          const r = c.getBoundingClientRect();
          if (r.width < 2 || r.height < 2) continue;
          try { c.scrollIntoView({ block: 'center', behavior: 'instant' }); } catch (e) {}
          c.click();
          return true;
        }
        return false;
      };
      return new Promise(resolve => {
        let tries = 0;
        const tick = () => {
          if (pickAndClick())
            return setTimeout(() => resolve(snapshot('Кликнул по «' + req.selector + '».')), 800);
          if (++tries >= 10) return resolve('Не нашёл элемент по селектору за 5с: ' + req.selector);
          setTimeout(tick, 500);
        };
        tick();
      });
    }
    if (req.action === 'clicktext') {
      // Клик по элементу с заданным ВИДИМЫМ ТЕКСТОМ — достаёт то, что структурный снапшот не
      // пронумеровал (динамические дропдауны/оверлеи SPA: результаты autocomplete и пр.).
      // Общий приём, не пер-сайт. Ждём появления (SPA дорисовывают) до ~5с.
      const want = (req.text || '').trim().toLowerCase();
      if (!want) return 'Не указан текст для клика.';
      // Ближайший КЛИКАБЕЛЬНЫЙ предок (или сам) — то, по чему реально навигируют (ссылка/кнопка).
      const clickableOf = (el) => {
        for (let n = el, hops = 0; n && hops < 5; n = n.parentElement, hops++) {
          const tag = n.tagName ? n.tagName.toLowerCase() : '';
          if (tag === 'a' || tag === 'button' || (n.getAttribute && n.getAttribute('role') === 'button')
              || (n.hasAttribute && n.hasAttribute('onclick'))) return n;
          try { if (getComputedStyle(n).cursor === 'pointer') return n; } catch (e) {}
        }
        return null;
      };
      const findOne = () => {
        let best = null, bestLen = Infinity;
        walk(document.documentElement, document, false, (el) => {
          if (!el.getBoundingClientRect) return;
          const tag = el.tagName ? el.tagName.toLowerCase() : '';
          if (tag === 'input' || tag === 'textarea' || tag === 'select') return;  // не строка ввода
          const r = el.getBoundingClientRect();
          if (r.width < 2 || r.height < 2) return;
          let s; try { s = getComputedStyle(el); } catch (e) { return; }
          if (s.visibility === 'hidden' || s.display === 'none' || s.opacity === '0') return;
          const txt = (el.innerText || el.textContent || el.getAttribute('aria-label') || '')
            .trim().toLowerCase();
          if (!txt || !(txt === want || txt.includes(want))) return;
          // Целимся только в то, что реально кликабельно (ссылка/кнопка/pointer) — иначе клик
          // по голому span ничего не делает (живой провал: anilibria-дропдаун навигирует <a>).
          const target = clickableOf(el);
          if (!target) return;
          if (txt.length < bestLen) { best = target; bestLen = txt.length; }
        });
        return best;
      };
      return new Promise(resolve => {
        let tries = 0;
        const tick = () => {
          const target = findOne();  // уже кликабельный (ссылка/кнопка/pointer)
          if (target) {
            const pl = payLabel(target);
            if (pl) return resolve(payRefusal(pl));  // [ГРАНИЦА ОПЛАТЫ]
            try { target.scrollIntoView({ block: 'center', behavior: 'instant' }); } catch (e) {}
            target.click();
            return setTimeout(() => resolve(snapshot('Кликнул по тексту «' + req.text + '».')), 900);
          }
          if (++tries >= 10) return resolve('Не нашёл на странице кликабельный элемент с текстом «' + req.text + '» — сделай see/read.');
          setTimeout(tick, 500);
        };
        tick();
      });
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
      // PAUSE без медиа-элементов (Я.Музыка играет через Web Audio, <audio> нет) →
      // жмём видимую кнопку «Пауза» (общая эвристика кастомных плееров).
      if ((req.mediaAction === 'pause' || req.mediaAction === 'toggle') && n === 0 && isPlaying()) {
        const reP = /пауза|pause|приостанов/i;
        for (const c of document.querySelectorAll('button,[role="button"],[aria-label],[title]')) {
          const r = c.getBoundingClientRect();
          if (r.width < 1 || r.height < 1) continue;
          const lbl = (c.getAttribute('aria-label')||c.getAttribute('title')||c.textContent||'').trim();
          if (reP.test(lbl)) { c.click(); n++; break; }
        }
      }
      // PLAY без играющего звука → жмём кнопку воспроизведения. ВАЖНО: сначала КОНТЕКСТНАЯ
      // кнопка запуска («Слушать»/«Listen»/«Play all/Play») — она грузит НОВЫЙ контекст
      // (артист/плейлист/избранное), а НЕ глобальный плеер внизу, который возобновил бы СТАРЫЙ
      // залипший трек (живой баг: «включи избранное» → играл прошлый трек). Только если такой
      // нет — общая эвристика play (одиночный трек/видео).
      if ((req.mediaAction === 'play' || req.mediaAction === 'toggle') && !isPlaying()) {
        const cands = [...document.querySelectorAll(
          'button,[role="button"],a,[aria-label],[title],[data-test-id*="play" i]')];
        const labelOf = (el) => (el.getAttribute('aria-label')||el.getAttribute('title')||
                         el.textContent||el.getAttribute('data-test-id')||'').trim();
        const vis = (el) => { const r = el.getBoundingClientRect(); return r.width >= 1 && r.height >= 1; };
        // 1) КОНТЕКСТ-запуск: «Слушать»/«Listen»/«Play all/Слушать всё/Воспроизвести всё»
        const reCtx = /^(слушать|слушать всё|listen|play|play all|воспроизвести всё|играть)$/i;
        let clicked = false;
        for (const el of cands) {
          if (!vis(el)) continue;
          if (reCtx.test(labelOf(el))) { el.click(); n++; clicked = true; break; }
        }
        // 2) Фолбэк: общая эвристика (одиночный трек/видео — нет контекст-кнопки)
        if (!clicked) {
          const re = /\bplay\b|слуша|воспроизв|▶/i;
          for (const el of cands) {
            if (!vis(el)) continue;
            if (re.test(labelOf(el))) { el.click(); n++; break; }
          }
        }
      }
      // Медиа стартует не мгновенно — подождём ~1.5с перед вердиктом (по isPlaying).
      const wantQuiet = ['pause', 'mute'].includes(req.mediaAction);
      return new Promise(resolve => setTimeout(() => resolve(
        `${req.mediaAction}: действий ${n}; ` +
        (isPlaying() ? `♪ ЗВУК ИГРАЕТ (${nowTitle()})`
         : wantQuiet ? '⏸ тихо (готово)'
                     : 'звук не пошёл — кликни конкретную кнопку плей из снапшота')
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
    if (req.action === 'locateplay') {
      // Координаты лучшей кнопки воспроизведения (для TRUSTED-клика через CDP из background:
      // обычный el.click() не даёт user activation → Chrome блокирует autoplay звука).
      const re = /\bplay\b|слуша|воспроизв|▶/i;
      let best = null, bestArea = 0;
      for (const c of document.querySelectorAll('button,[role="button"],a,[aria-label],[title],[data-test-id*="play" i]')) {
        const r = c.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) continue;
        const label = (c.getAttribute('aria-label')||c.getAttribute('title')||c.textContent||c.getAttribute('data-test-id')||'').trim();
        if (!re.test(label)) continue;
        const area = r.width * r.height;   // главная кнопка «Слушать» обычно крупнее иконок в строках
        if (area > bestArea) { best = c; bestArea = area; }
      }
      if (!best) return '';
      try { best.scrollIntoView({ block: 'center', behavior: 'instant' }); } catch (e) {}
      const r = best.getBoundingClientRect();
      return Math.round(r.left + r.width / 2) + ',' + Math.round(r.top + r.height / 2);
    }
    if (req.action === 'coordnum') {
      // Центр элемента по НОМЕРУ из снапшота — для TRUSTED-клика (React-кнопки игнорят
      // обычный el.click(): нужен настоящий жест через CDP). Сначала прокрутить к нему.
      const e = req.item != null ? findIndexed(req.item) : null;
      if (!e) return '';
      try { e.scrollIntoView({ block: 'center', behavior: 'instant' }); } catch (x) {}
      const r = e.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) return '';
      return Math.round(r.left + r.width / 2) + ',' + Math.round(r.top + r.height / 2);
    }
    if (req.action === 'locatevideo') {
      // Координаты центра САМОГО КРУПНОГО видимого <video> — для TRUSTED-клика по плееру.
      // Общий приём для кастомных плееров (jut.su и пр.), которые грузят источник ТОЛЬКО по
      // клику на сам плеер-оверлей (нет семантической кнопки) — не хардкод сайтов.
      let best = null, bestArea = 0;
      for (const v of document.querySelectorAll('video')) {
        const r = v.getBoundingClientRect();
        if (r.width < 80 || r.height < 60) continue;  // не считаем крошечные/превью
        const area = r.width * r.height;
        if (area > bestArea) { best = v; bestArea = area; }
      }
      if (!best) return '';
      try { best.scrollIntoView({ block: 'center', behavior: 'instant' }); } catch (e) {}
      const r = best.getBoundingClientRect();
      return Math.round(r.left + r.width / 2) + ',' + Math.round(r.top + r.height / 2);
    }
    if (req.action === 'playing') {
      // Вердикт о звуке (после trusted-клика): формат совпадает с media — act_node его понимает.
      return new Promise(resolve => setTimeout(() => resolve(
        isPlaying() ? `♪ ЗВУК ИГРАЕТ (${nowTitle()}) · ${location.href}`
                    : 'звук не пошёл — кликни конкретную кнопку плей из снапшота'
      ), 2000));
    }
    if (req.action === 'probearm') {
      // Диагностика: поймать СЛЕДУЮЩИЙ клик по странице (долетел ли CDP-клик и куда).
      globalThis.__agentProbe = null;
      document.addEventListener('click', (e) => {
        globalThis.__agentProbe = `click x=${e.clientX} y=${e.clientY} trusted=${e.isTrusted} ` +
          `target=${e.target.tagName}:${((e.target.getAttribute && e.target.getAttribute('aria-label')) || e.target.textContent || '').trim().slice(0, 40)}`;
      }, { once: true, capture: true });
      return 'armed';
    }
    if (req.action === 'proberead') return String(globalThis.__agentProbe);
    if (req.action === 'mediainfo') {
      const m = document.querySelector('video,audio');
      if (!m) return 'нет медиа-элемента';
      return JSON.stringify({ paused: m.paused, muted: m.muted, volume: m.volume,
        t: +m.currentTime.toFixed(1), dur: m.duration, readyState: m.readyState,
        networkState: m.networkState, vis: document.visibilityState });
    }
    if (req.action === 'diag') {
      // Диагностика плеера: какие кнопки матчатся под play-эвристику, есть ли медиа-элементы,
      // что говорит mediaSession, и чем падает прямой .play() (autoplay-policy и т.п.).
      const re = /\bplay\b|слуша|воспроизв|▶/i;
      const matched = [];
      for (const el of document.querySelectorAll('button,[role="button"],a,[aria-label],[title],[data-test-id*="play" i]')) {
        const r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) continue;
        const label = (el.getAttribute('aria-label')||el.getAttribute('title')||el.textContent||el.getAttribute('data-test-id')||'').trim();
        if (re.test(label)) matched.push(el.tagName + ': ' + label.slice(0, 60));
        if (matched.length >= 8) break;
      }
      const ms = [...document.querySelectorAll('audio,video')];
      const states = ms.slice(0, 4).map(m => `${m.tagName} paused=${m.paused} t=${m.currentTime.toFixed(1)} src=${(m.currentSrc||'').slice(0,60)}`);
      let sess = 'нет';
      try { sess = navigator.mediaSession ? String(navigator.mediaSession.playbackState) : 'нет'; } catch (e) {}
      return new Promise(resolve => {
        const fin = (playErr) => resolve(
          `видимость: ${document.visibilityState} focus=${document.hasFocus()}\n` +
          `активация: ${navigator.userActivation ? 'active=' + navigator.userActivation.isActive + ' hasBeen=' + navigator.userActivation.hasBeenActive : '?'}\n` +
          `mediaSession: ${sess}\nмедиа-элементов: ${ms.length}\n${states.join('\n')}\n` +
          `прямой play(): ${playErr}\nplay-кнопки под эвристику:\n${matched.join('\n') || '(нет)'}`);
        if (!ms.length) return fin('(нет элементов)');
        ms[0].play().then(() => fin('OK — пошёл')).catch(e => fin(e.name + ': ' + e.message));
      });
    }
    return 'Неизвестное действие.';
  } catch (e) { return 'Ошибка в странице: ' + e.message; }
};
