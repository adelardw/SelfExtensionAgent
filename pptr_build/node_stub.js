// CJS-стаб node-билтина: ЛЮБОЙ именованный импорт → no-op. Node-пути puppeteer-core
// (лаунчеры/ScreenRecorder/файлы/сеть) в расширении не исполняются — ExtensionTransport
// общается через chrome.debugger. Proxy удовлетворяет статический импорт esbuild (CJS-interop).
const noop = function () {};
const handler = { get: () => noop, apply: () => undefined, construct: () => ({}) };
module.exports = new Proxy(noop, handler);
