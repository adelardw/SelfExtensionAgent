// Service worker: держит WebSocket к локальному агенту (127.0.0.1:8777), принимает
// команды, исполняет их в активной/целевой вкладке ПОЛЬЗОВАТЕЛЯ через chrome.scripting.
// Токен берётся из storage (его кладёт popup при первой настройке).
const BRIDGE = "ws://127.0.0.1:8777";
let ws = null;
let agentTabId = null;  // вкладка, в которой агент работает (фон — не крадёт фокус)

function setBadge(ok) {
  chrome.action.setBadgeText({ text: ok ? "●" : "" });
  chrome.action.setBadgeBackgroundColor({ color: ok ? "#2da44e" : "#999" });
}

async function getToken() {
  const { agentToken } = await chrome.storage.local.get("agentToken");
  return agentToken || "";
}

function waitTabLoaded(tabId, timeout = 8000) {
  // Ждём РЕАЛЬНОЙ загрузки вкладки (а не фиксированную паузу — раньше всегда 1.5с тормозили).
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; chrome.tabs.onUpdated.removeListener(onUpd); resolve(); } };
    const onUpd = (id, info) => { if (id === tabId && info.status === "complete") finish(); };
    chrome.tabs.onUpdated.addListener(onUpd);
    chrome.tabs.get(tabId, (t) => { if (t && t.status === "complete") finish(); });
    setTimeout(finish, timeout);
  });
}

// id вкладки агента ПЕРЕЖИВАЕТ перезапуск service worker (MV3 убивает SW и обнуляет
// переменные → раньше каждый open плодил НОВУЮ вкладку). Храним в session storage.
async function getAgentTab() {
  if (agentTabId !== null) return agentTabId;
  try {
    const { agentTabId: saved } = await chrome.storage.session.get("agentTabId");
    if (saved != null) agentTabId = saved;
  } catch (e) {}
  return agentTabId;
}
async function setAgentTab(id) {
  agentTabId = id;
  try { await chrome.storage.session.set({ agentTabId: id }); } catch (e) {}
}

async function ensureTab(url) {
  // ОДНА фоновая вкладка агента (active:false → не дёргает пользователя, переиспользуется).
  let id = await getAgentTab();
  if (id !== null) {
    try { await chrome.tabs.get(id); }   // жива ли (могли закрыть)
    catch { id = null; await setAgentTab(null); }
  }
  if (id === null) {
    // Отдельная фоновая ВКЛАДКА в окне юзера (его выбор: вкладка, не окно). Для старта
    // звука она ненадолго активируется и потом возвращается прежняя (см. wantSound).
    let tab;
    try {
      tab = await chrome.tabs.create({ url: url || "about:blank", active: false });
    } catch (e) {
      // «No current window»: Chrome фоном без окон (open -g на macOS) — создаём окно.
      const win = await chrome.windows.create({ url: url || "about:blank", focused: false });
      tab = win.tabs[0];
    }
    await setAgentTab(tab.id);
    if (url) await waitTabLoaded(tab.id);
    return tab.id;
  }
  if (url) {
    await chrome.tabs.update(id, { url, active: false });  // НЕ активируем — фон
    await waitTabLoaded(id);
  }
  return id;
}

async function runInTab(tabId, req) {
  const [res] = await chrome.scripting.executeScript({
    target: { tabId },
    func: (r) => agentExec(r),     // agentExec вставляется отдельным файлом ниже
    args: [req],
  });
  return res?.result ?? "(нет результата)";
}

async function cdpClick(target, x, y) {
  // Полная последовательность как у Puppeteer (moved→pressed→released, buttons:1):
  // только так Chrome засчитывает «жест пользователя» и снимает autoplay-блок.
  await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent",
    { type: "mouseMoved", x, y, button: "none", buttons: 0, pointerType: "mouse" });
  await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent",
    { type: "mousePressed", x, y, button: "left", buttons: 1, clickCount: 1, pointerType: "mouse" });
  await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent",
    { type: "mouseReleased", x, y, button: "left", buttons: 1, clickCount: 1, pointerType: "mouse" });
}

// Клик по плею В КОНТЕКСТЕ СТРАНИЦЫ с userGesture:true (Runtime.evaluate через CDP):
// даёт user activation, без которой Chrome блокирует autoplay звука в фоновой вкладке
// (живой баг: «Воспроизведение» кликается, но звук не идёт — медиа-элемент не создаётся).
// Без координат → не страдает от сдвига страницы инфо-полоской «отлаживает браузер».
const PLAY_NUDGE_JS = `(() => {
  const re = /\\bplay\\b|слуша|воспроизв|▶/i;  // play\\b: НЕ матчить "Playlist" (уводил в плейлист)
  let best = null, bestArea = 0;
  for (const c of document.querySelectorAll('button,[role="button"],a,[aria-label],[title],[data-test-id*="play" i]')) {
    const r = c.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const label = (c.getAttribute('aria-label')||c.getAttribute('title')||c.textContent||c.getAttribute('data-test-id')||'').trim();
    if (!re.test(label)) continue;
    const area = r.width * r.height;
    if (area > bestArea) { best = c; bestArea = area; }
  }
  const m = document.querySelector('audio,video');
  if (m) m.play().catch(() => {});
  if (best) { best.click(); return 'clicked: ' + (best.getAttribute('aria-label')||best.textContent||'').trim().slice(0,40); }
  return m ? 'play() на медиа' : 'кнопка не найдена';
})()`;

async function trustedSoundNudge(tabId) {
  // Подход как у nanobrowser/puppeteer-in-extension: НАСТОЯЩИЙ ввод через CDP по координатам
  // (трастед-события — нужны React-плеерам, слушающим pointer*/isTrusted), затем бэкап —
  // gesture-evaluate. Координаты снимаем ПОСЛЕ attach (инфо-полоска дебага сдвигает страницу).
  // Вердикт о звуке считаем ДО detach: отключение может отменить недоигранный ввод.
  const target = { tabId };
  await chrome.debugger.attach(target, "1.3");
  try {
    await new Promise(r => setTimeout(r, 300));  // перевёрстка под инфо-полоску
    let how = "";
    const loc = await runInTab(tabId, { action: "locateplay" });
    if (typeof loc === "string" && loc.includes(",")) {
      const [x, y] = loc.split(",").map(Number);
      await cdpClick(target, x, y);
      how = `cdp-клик (${x},${y})`;
    }
    let verdict = await runInTab(tabId, { action: "playing" });  // внутри ждёт ~2с
    if (typeof verdict === "string" && verdict.includes("звук не пошёл")) {
      const ev = await chrome.debugger.sendCommand(target, "Runtime.evaluate",
        { expression: PLAY_NUDGE_JS, userGesture: true, returnByValue: true });
      how += " → gesture-клик: " + (ev && ev.result ? String(ev.result.value) : "?");
      verdict = await runInTab(tabId, { action: "playing" });
    }
    // ФИНАЛЬНЫЙ ФОЛБЭК — клик по САМОМУ ВИДЕО: кастомные плееры (jut.su и пр.) грузят источник
    // только по клику на плеер-оверлей (нет семантической кнопки). Trusted-клик по центру
    // <video>. Срабатывает ТОЛЬКО если звук всё ещё не пошёл → музыку/готовые плееры не трогает.
    if (typeof verdict === "string" && verdict.includes("звук не пошёл")) {
      const vloc = await runInTab(tabId, { action: "locatevideo" });
      if (typeof vloc === "string" && vloc.includes(",")) {
        const [vx, vy] = vloc.split(",").map(Number);
        await cdpClick(target, vx, vy);
        how += ` → клик по видео (${vx},${vy})`;
        verdict = await runInTab(tabId, { action: "playing" });
      }
    }
    return verdict + (how ? " · " + how : "");
  } finally {
    try { await chrome.debugger.detach(target); } catch (e) {}
  }
}

async function handle(msg) {
  const a = msg.action, args = msg.args || {};
  if (a === "ver") return "bg-v12 (video-click)";  // диагностика: какой воркер реально жив
  if (a === "tclick") {  // диагностика: trusted-клик по явным координатам
    const tid = await ensureTab(null);
    const target = { tabId: tid };
    await chrome.debugger.attach(target, "1.3");
    try { await cdpClick(target, Number(args.x), Number(args.y)); return "tclick ok"; }
    finally { try { await chrome.debugger.detach(target); } catch (e) {} }
  }
  if (a === "open") {
    const tabId = await ensureTab(args.url.startsWith("http") ? args.url : "https://" + args.url);
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content_actions.js"] });
    return await runInTab(tabId, { action: "see", note: "Открыл " + args.url });
  }
  const tabId = await ensureTab(null);
  // СТАРТ ЗВУКА ТРЕБУЕТ ВИДИМОСТИ: в скрытой странице (visibilityState=hidden) веб-плееры
  // не инициализируются и Chrome отсекает старт звука (живой диагноз: «видимость: hidden»,
  // клики доходят, активация есть, а <audio> не создаётся). Делаем вкладку агента АКТИВНОЙ
  // в её окне (окно может оставаться позади приложений юзера — фокус ОС не крадём);
  // разворачивание самого окна делает мост (osascript) на стороне агента.
  const wantSound = a === "media" && ["play", "toggle"].includes(args.action);
  let prevWinId = null, prevTabId = null;  // куда вернуть юзера после старта звука
  if (wantSound) {
    try {
      const t = await chrome.tabs.get(tabId);
      const prev = await chrome.windows.getLastFocused();
      if (prev && prev.id !== t.windowId && prev.focused) prevWinId = prev.id;
      const [actTab] = await chrome.tabs.query({ active: true, windowId: t.windowId });
      if (actTab && actTab.id !== tabId) prevTabId = actTab.id;
      // БЕЗ спуфа Visibility (он рвал Spotify-DRM — у юзера фон играет сам). Просто на миг
      // делаем вкладку видимой, чтобы плеер инициализировался и autoplay-policy пустила звук;
      // дальше Spotify сам держит фоновое воспроизведение (как при ручном переключении).
      await chrome.tabs.update(tabId, { active: true });
      await chrome.windows.update(t.windowId, { focused: true, state: "normal" });
      // Spotify дольше поднимает плеер (EME/DRM), чем ЯМ → даём больше времени до клика,
      // иначе кликаем по неинициализированному плееру и звук не стартует.
      await new Promise(r => setTimeout(r, 1200));
    } catch (e) {}
  }
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content_actions.js"] });
  const req = { action: a, item: args.item, text: args.text, submit: args.submit,
                key: args.key, direction: args.direction, mediaAction: args.action,
                selector: args.selector };
  let result = await runInTab(tabId, req);
  // ДОЖИМ ЗВУКА: play не дал звука → DOM-клик не считается жестом пользователя
  // (autoplay-policy). Находим кнопку плея и кликаем НАСТОЯЩИМ кликом через CDP.
  if (wantSound && typeof result === "string" && result.includes("звук не пошёл")) {
    try {
      const verdict = await trustedSoundNudge(tabId);
      if (verdict) result = verdict;
    } catch (e) { result += " (trusted-клик не вышел: " + e.message + ")"; }
  }
  // Дать звуку устаканиться, прежде чем уводить вкладку в фон (Spotify иначе может
  // встать на первой же секунде). Возврат фокуса — НЕ пауза: юзер подтвердил, что фон играет.
  if (wantSound) await new Promise(r => setTimeout(r, 700));
  // Вернуть юзера, откуда забрали (звук уже идёт; не выдёргиваем с его сайта):
  // его прежнюю ВКЛАДКУ в этом окне и его прежнее ОКНО, если фокус был в другом.
  if (prevTabId !== null) {
    try { await chrome.tabs.update(prevTabId, { active: true }); } catch (e) {}
  }
  if (prevWinId !== null) {
    try { await chrome.windows.update(prevWinId, { focused: true }); } catch (e) {}
  }
  return result;
}

const chatPending = new Map();  // id -> resolve (чат-запросы из side panel)

function wsReady() { return ws && ws.readyState === 1; }

function connect() {
  ws = new WebSocket(BRIDGE);
  ws.onopen = async () => { ws.send(JSON.stringify({ token: await getToken() })); setBadge(true); };
  ws.onclose = () => { setBadge(false); setTimeout(connect, 3000); };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
  ws.onmessage = async (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    // Ответ на наш чат-запрос (extension→agent)?
    if (msg.id && chatPending.has(msg.id)) {
      chatPending.get(msg.id)(msg.result);
      chatPending.delete(msg.id);
      return;
    }
    // Иначе это команда агента (agent→extension): исполнить и ответить.
    let result;
    try { result = await handle(msg); }
    catch (e) { result = "Ошибка расширения: " + e.message; }
    ws.send(JSON.stringify({ id: msg.id, result }));
  };
}

// Чат из side panel: шлём агенту, ждём ответ, возвращаем в панель.
chrome.runtime.onMessage.addListener((req, _sender, sendResponse) => {
  if (req.type === "agentChat") {
    if (!wsReady()) { sendResponse({ ok: false, error: "нет связи с агентом" }); return true; }
    const id = "chat-" + Math.random().toString(36).slice(2);
    chatPending.set(id, (result) => sendResponse({ ok: true, result }));
    ws.send(JSON.stringify({ type: "chat", id, text: req.text }));
    setTimeout(() => {
      if (chatPending.has(id)) { chatPending.delete(id); sendResponse({ ok: false, error: "таймаут" }); }
    }, 120000);
    return true;  // async sendResponse
  }
  if (req.type === "agentStatus") { sendResponse({ connected: wsReady() }); return true; }
});

// Клик по иконке открывает side-panel (чат).
chrome.action.onClicked?.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
});

// KEEPALIVE: в Manifest V3 service worker засыпает через ~30с простоя и убивает WS
// (→ агент видел «не подключено»). Будильник каждые ~24с оживляет worker, шлёт ping и
// переподключает WS, если он закрылся. Так связь держится постоянно.
chrome.alarms.create("keepalive", { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener((a) => {
  if (a.name !== "keepalive") return;
  if (wsReady()) { try { ws.send(JSON.stringify({ type: "ping" })); } catch (e) {} }
  else connect();
});

connect();
chrome.runtime.onStartup?.addListener(connect);
chrome.runtime.onInstalled?.addListener(connect);
