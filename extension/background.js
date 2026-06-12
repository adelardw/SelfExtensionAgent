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

async function ensureTab(url) {
  // Работаем в ОТДЕЛЬНОЙ вкладке агента (active:false → фон, не дёргает пользователя).
  if (agentTabId !== null) {
    try { await chrome.tabs.get(agentTabId); }
    catch { agentTabId = null; }
  }
  if (agentTabId === null) {
    const tab = await chrome.tabs.create({ url: url || "about:blank", active: false });
    agentTabId = tab.id;
    if (url) await waitTabLoaded(agentTabId);
  } else if (url) {
    await chrome.tabs.update(agentTabId, { url, active: false });
    await waitTabLoaded(agentTabId);
  }
  return agentTabId;
}

async function runInTab(tabId, req) {
  const [res] = await chrome.scripting.executeScript({
    target: { tabId },
    func: (r) => agentExec(r),     // agentExec вставляется отдельным файлом ниже
    args: [req],
  });
  return res?.result ?? "(нет результата)";
}

async function handle(msg) {
  const a = msg.action, args = msg.args || {};
  if (a === "open") {
    const tabId = await ensureTab(args.url.startsWith("http") ? args.url : "https://" + args.url);
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content_actions.js"] });
    return await runInTab(tabId, { action: "see", note: "Открыл " + args.url });
  }
  const tabId = await ensureTab(null);
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content_actions.js"] });
  const req = { action: a, item: args.item, text: args.text, submit: args.submit,
                key: args.key, direction: args.direction, mediaAction: args.action };
  return await runInTab(tabId, req);
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
