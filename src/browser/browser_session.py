"""
Живой управляемый браузер агента (контур «агент может делать в браузере ВСЁ»).

Принципы:
- СТРУКТУРНОЕ управление: страница «видится» нумерованным списком интерактивных
  элементов (DOM), клик/ввод — по номеру. Не слепые клавиши по скриншоту и не
  Accessibility-права macOS.
- ВИДИМОЕ окно (headless=False) + ПОСТОЯННЫЙ профиль (data/browser_profile):
  логины (Яндекс Музыка, YouTube, …) переживают сессии — один раз вошёл, дальше
  агент действует под тобой.
- Один singleton на процесс. Sync-API playwright (greenlet) привязан к потоку —
  ВСЕ операции идут через выделенный однопоточный executor. Браузер живёт между
  запросами графа (модульный _state) → «поставь паузу» позже бьёт в то же окно.
- Навык browser_control перезагружается на каждом подключении тулов (exec_module) —
  поэтому живое состояние лежит ЗДЕСЬ, в core-модуле, а не в навыке.
"""
from __future__ import annotations

import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Профиль агента в РЕАЛЬНОМ системном Chrome (отдельный user-data-dir → не конфликтует
# с основным Chrome юзера, работает хоть тот открыт, хоть закрыт). Логины (Яндекс Музыка,
# YouTube, …) сохраняются здесь: вошёл один раз — дальше агент действует под тобой.
PROFILE_DIR = Path("data/chrome_agent_profile")
OP_TIMEOUT = 60  # секунд на операцию (включая первый запуск Chrome)

# СТРУКТУРНЫЙ сентинел воспроизведения: in-repo путь ЗНАЕТ ground-truth (!m.paused), поэтому
# эмитит фиксированный машинный токен, а не прозу. act_node читает токен (детерминированно), без
# регэкспа по естественному языку (раньше парсил «ЗВУК ИГРАЕТ» строкой — хрупкая связка с форматом).
MEDIA_PLAYING = "[[MEDIA_PLAYING]]"

_EX = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-browser")
# pw — хэндл playwright; channel — 'chrome' (системный) или None (фолбэк-Chromium).
_state: dict = {"ctx": None, "page": None, "pw": None}

# JS: пометить видимые интерактивные элементы data-agent-i и вернуть их описания.
_SNAPSHOT_JS = """
() => {
  const sel = 'a, button, input, textarea, select, [role="button"], [role="link"], ' +
              '[role="tab"], [role="menuitem"], [role="checkbox"], [onclick], ' +
              '[contenteditable="true"], audio, video, [role="searchbox"]';
  const vis = el => { const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1 && r.bottom > 0 && r.top < innerHeight * 1.5; };
  // Срезаем ЗАРЕЗЕРВИРОВАННЫЕ структурные токены [[MEDIA_…]] из текста ЭЛЕМЕНТА страницы: иначе
  // страница с aria-label «[[MEDIA_PLAYING]] звук играет» подделала бы наш ground-truth-сигнал
  // (баг ревью NEW-3). Маркер ниже эмитим ТОЛЬКО мы из media.some(!m.paused).
  const txt = el => (el.getAttribute('aria-label') || el.innerText || el.value ||
                     el.getAttribute('placeholder') || el.getAttribute('title') || '')
                    .replace(/\\[\\[MEDIA_[^\\]]*\\]\\]/gi, '')
                    .trim().replace(/\\s+/g, ' ').slice(0, 80);
  const out = [];
  let i = 0;
  for (const el of document.querySelectorAll(sel)) {
    if (!vis(el)) continue;
    el.setAttribute('data-agent-i', String(i));
    const tag = el.tagName.toLowerCase();
    let kind = tag === 'input' ? `input:${el.type || 'text'}` : tag;
    if (tag === 'audio' || tag === 'video')
      kind += el.paused ? ' (на паузе)' : ' (СЕЙЧАС ИГРАЕТ)';
    out.push({i, kind, text: txt(el)});
    if (++i >= 60) break;
  }
  // Состояние воспроизведения видно агенту ДАЖЕ если плеер за пределами снапшота.
  const media = [...document.querySelectorAll('audio, video')];
  if (media.some(m => !m.paused))
    out.unshift({i: -1, kind: '♪ медиа', text: '[[MEDIA_PLAYING]] звук играет — цель достигнута, не перепроверяй'});
  return out.filter(o => o.i >= 0 || o.kind === '♪ медиа');
}
"""


def _hidden() -> bool:
    """По умолчанию окно ВИДИМОЕ (фоновая среда юзера). Скрытно (headless) — осознанная фича."""
    try:
        from src.config.cli_config import get_cli
        return bool(get_cli("browser_hidden"))
    except Exception:  # noqa: BLE001
        return False


def _background() -> bool:
    """ФОНОВЫЙ режим (по умолчанию ВКЛ): окно агента НЕ крадёт фокус — пользователь
    продолжает свою работу, музыка играет фоном, переключится сам. Отключить:
    config cli.browser_background: false."""
    try:
        from src.config.cli_config import get_cli
        v = get_cli("browser_background")
        return True if v is None else bool(v)
    except Exception:  # noqa: BLE001
        return True


# Какое приложение было активным ДО запуска браузера — чтобы вернуть фокус ему (macOS).
_prev_app: str | None = None


def _capture_front():
    global _prev_app
    if platform.system() != "Darwin" or not _background():
        return
    try:
        _prev_app = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to name of first process whose frontmost is true'],
            capture_output=True, text=True, timeout=3).stdout.strip() or None
    except Exception:  # noqa: BLE001
        _prev_app = None


def _restore_front():
    """Вернуть фокус приложению пользователя — окно браузера остаётся в фоне (macOS)."""
    if platform.system() != "Darwin" or not _background() or not _prev_app:
        return
    try:
        subprocess.run(["osascript", "-e", f'tell application "{_prev_app}" to activate'],
                       capture_output=True, timeout=3)
    except Exception:  # noqa: BLE001
        pass


def _open_context(pw, channel):
    """launch_persistent_context системного Chrome (channel='chrome') или фолбэк-Chromium.
    Маскируем автоматизацию: Google по флагу --enable-automation душит сервисы и
    РАЗЛОГИНИВАЕТ (живой баг с YouTube). Убираем его и automation-признак в JS."""
    kw = dict(
        user_data_dir=str(PROFILE_DIR), headless=_hidden(),
        ignore_default_args=["--enable-automation"],  # ← главный признак, по которому банят
        args=["--no-first-run", "--no-default-browser-check",
              "--disable-blink-features=AutomationControlled"],
    )
    if channel:
        kw["channel"] = channel
    ctx = pw.chromium.launch_persistent_context(**kw)
    try:  # navigator.webdriver=false для всех страниц контекста (анти-детект)
        ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    except Exception:  # noqa: BLE001
        pass
    return ctx


def _ensure_page():
    """Поднять браузер (в потоке executor'а): РЕАЛЬНЫЙ системный Chrome (channel='chrome')
    с постоянным профилем агента — работает независимо от основного Chrome юзера. Нет
    Chrome в системе → фолбэк на бандл-Chromium. config cli.browser_backend:
    'system' (по умолч.) | 'chromium'."""
    page = _state.get("page")
    if page is not None:
        try:
            _ = page.title()  # жив ли (окно могли закрыть руками)
            return page
        except Exception:  # noqa: BLE001
            _state["ctx"] = _state["page"] = None

    backend = "system"
    try:
        from src.config.cli_config import get_cli
        backend = (get_cli("browser_backend") or "system").lower()
    except Exception:  # noqa: BLE001
        pass

    from playwright.sync_api import sync_playwright
    pw = _state.get("pw") or sync_playwright().start()
    _state["pw"] = pw

    _capture_front()  # запомнить активное приложение юзера ДО того, как Chrome заберёт фокус
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    ctx = None
    if backend != "chromium":
        try:
            ctx = _open_context(pw, channel="chrome")  # настоящий Google Chrome
        except Exception as e:  # noqa: BLE001
            print(f"[browser] системный Chrome недоступен ({type(e).__name__}) → Chromium")
    if ctx is None:
        ctx = _open_context(pw, channel=None)  # бандл-Chromium playwright
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    _state["ctx"], _state["page"] = ctx, page
    _restore_front()  # окно поднялось — сразу отдаём фокус обратно работе пользователя
    return page


def _call(fn):
    """Все операции — в ЕДИНСТВЕННОМ потоке браузера (sync-API не потокобезопасен).
    После операции в фоновом режиме возвращаем фокус приложению пользователя."""
    def _wrapped():
        try:
            return fn()
        finally:
            _restore_front()
    return _EX.submit(_wrapped).result(timeout=OP_TIMEOUT)


def _snapshot(page, note: str = "") -> str:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=10_000)
    except Exception:  # noqa: BLE001
        pass
    items = page.evaluate(_SNAPSHOT_JS) or []
    lines = [f"Страница: {page.title()!r} · {page.url}"]
    if note:
        lines.insert(0, note)
    lines += [(f"  ♪ {it['text']}" if it["i"] < 0 else
               f"  [{it['i']}] {it['kind']}: {it['text'] or '(без текста)'}") for it in items]
    if not items:
        lines.append("  (интерактивных элементов не видно — страница грузится? browser_see повторно)")
    lines.append("Дальше: browser_click(i) · browser_type(i, текст) · browser_scroll · browser_see")
    return "\n".join(lines)


# JS: поле персональных/платёжных данных? Агент в такие НЕ печатает (граница доверия).
_SENSITIVE_JS = """
(sel) => {
  const el = document.querySelector(sel);
  if (!el) return false;
  const s = ((el.type||'') + ' ' + (el.name||'') + ' ' + (el.id||'') + ' ' +
             (el.autocomplete||'') + ' ' + (el.placeholder||'')).toLowerCase();
  return el.type === 'password' ||
         /cc-|card|cvc|cvv|карт|cvc|паспорт|passport|снилс|snils|инн\\b/.test(s);
}
"""

# JS: управление ЛЮБЫМ медиа на странице (audio/video, включая под капотом веб-плееров).
_MEDIA_JS = """
(action) => {
  const ms = [...document.querySelectorAll('audio, video')];
  let n = 0;
  for (const m of ms) {
    if (action === 'pause' && !m.paused) { m.pause(); n++; }
    else if (action === 'play' && m.paused) { m.play(); n++; }
    else if (action === 'toggle') { m.paused ? m.play() : m.pause(); n++; }
    else if (action === 'mute' && !m.muted) { m.muted = true; n++; }
    else if (action === 'unmute' && m.muted) { m.muted = false; n++; }
  }
  return {total: ms.length, affected: n,
          playing: ms.filter(m => !m.paused).length};
}
"""


# ── Операции (зовутся из навыка browser_control) ────────────────────────────
def open_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    def _do():
        page = _ensure_page()
        # Повторное открытие ТОЙ ЖЕ страницы = перезагрузка → обрывает играющий трек/видео
        # (живой тест: музыка пошла и тут же перезапустилась). Уже там — только снапшот.
        if page.url.rstrip("/") == url.rstrip("/"):
            return _snapshot(page, note=f"Уже на {url} — НЕ перезагружаю (играющее не прерываю).")
        page.goto(url, timeout=45_000, wait_until="domcontentloaded")
        return _snapshot(page, note=f"Открыл {url}")
    return _call(_do)


def see() -> str:
    def _do():
        return _snapshot(_ensure_page())
    return _call(_do)


def click(item: int) -> str:
    def _do():
        page = _ensure_page()
        page.click(f'[data-agent-i="{int(item)}"]', timeout=8_000)
        page.wait_for_timeout(800)  # дать странице отреагировать (играть/перейти)
        return _snapshot(page, note=f"Кликнул [{int(item)}].")
    return _call(_do)


def type_into(item: int, text: str, submit: bool = True) -> str:
    def _do():
        page = _ensure_page()
        sel = f'[data-agent-i="{int(item)}"]'
        # Граница доверия: пароли/карты/документы агент НЕ заполняет — это поле юзера.
        try:
            if page.evaluate(_SENSITIVE_JS, sel):
                return ("[ГРАНИЦА ПЕРСОНАЛЬНЫХ ДАННЫХ] Поле похоже на пароль/карту/документ — "
                        "агент в такие НЕ печатает. Попроси пользователя (ask_user) заполнить "
                        "его самому в открытом окне и продолжай со СЛЕДУЮЩЕГО шага сценария.")
        except Exception:  # noqa: BLE001
            pass
        page.click(sel, timeout=8_000)
        page.fill(sel, text)
        if submit:
            page.press(sel, "Enter")
            page.wait_for_timeout(1200)
        return _snapshot(page, note=f"Ввёл «{text}» в [{int(item)}]" + (" и нажал Enter." if submit else "."))
    return _call(_do)


def press(key: str) -> str:
    def _do():
        page = _ensure_page()
        page.keyboard.press(key)
        page.wait_for_timeout(500)
        return _snapshot(page, note=f"Нажал {key}.")
    return _call(_do)


def scroll(direction: str = "down") -> str:
    def _do():
        page = _ensure_page()
        page.mouse.wheel(0, 700 if direction != "up" else -700)
        page.wait_for_timeout(400)
        return _snapshot(page, note=f"Проскроллил {direction}.")
    return _call(_do)


_READ_JS = """
() => {
  const drop = ['script','style','noscript','svg'];
  const cl = document.body ? document.body.cloneNode(true) : null;
  if (!cl) return '';
  for (const t of drop) cl.querySelectorAll(t).forEach(e => e.remove());
  return (cl.innerText || '').replace(/\\n{3,}/g, '\\n\\n').trim().slice(0, 6000);
}
"""


def read() -> str:
    """Видимый ТЕКСТ страницы (для «расскажи вкратце» / «порекомендуй»)."""
    def _do():
        page = _ensure_page()
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8_000)
        except Exception:  # noqa: BLE001
            pass
        txt = page.evaluate(_READ_JS) or ""
        head = f"Содержимое {page.title()!r} ({page.url}):\n"
        return head + (txt or "(текст не извлёкся — попробуй browser_see)")
    return _call(_do)


def media(action: str = "toggle") -> str:
    """Управление ЛЮБЫМ медиа на странице: pause | play | toggle | mute | unmute."""
    act = action if action in ("pause", "play", "toggle", "mute", "unmute") else "toggle"

    def _do():
        page = _ensure_page()
        res = page.evaluate(_MEDIA_JS, act) or {}
        if not res.get("total"):
            return ("Медиа-элементов на странице не найдено. Если плеер кастомный — "
                    "browser_see и клик по кнопке плеера.")
        playing = bool(res.get("playing"))
        mark = MEDIA_PLAYING + " " if playing else ""   # структурный флаг для act_node (не проза)
        state = "играет" if playing else "на паузе"
        return f"{mark}{act}: затронуто {res.get('affected', 0)} из {res['total']} медиа · сейчас {state}."
    return _call(_do)


def close() -> str:
    def _do():
        ctx = _state.get("ctx")
        _state["ctx"] = _state["page"] = None
        if ctx is not None:
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass
        return "Окно агента закрыто."
    return _call(_do)
