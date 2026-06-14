import { useEffect, useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, ArrowUp, Star, X, PanelLeft, Settings, Paperclip, Mic, Square, FileText, ChevronDown, Check } from "lucide-react"

type Thread = { thread_id: string; title: string; favorite?: number }
type Msg = { role: "user" | "assistant"; content: string }
type Cfg = { provider?: string; base_url?: string; api_key_source?: string; active?: string;
  model?: string; code_model?: string; deep_model?: string; work_mode?: string; force_mode?: string }

const uid = (() => {
  let u = localStorage.getItem("agent_uid")
  if (!u) { u = "gui-" + Math.random().toString(36).slice(2, 10); localStorage.setItem("agent_uid", u) }
  return u
})()
const newId = () => (crypto.randomUUID ? crypto.randomUUID() : "t-" + Date.now())
const esc = (s: string) => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!))

function md(s: string): string {
  s = esc(s)
  s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, _l, c) => "<pre><code>" + c.replace(/\n$/, "") + "</code></pre>")
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>")
  s = s.replace(/^### (.*)$/gm, "<h3>$1</h3>").replace(/^## (.*)$/gm, "<h2>$1</h2>").replace(/^# (.*)$/gm, "<h1>$1</h1>")
  s = s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>")
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  // авто-ссылки на «голые» URL (не трогаем те, что уже внутри href="…")
  s = s.replace(/(?<!["(])(https?:\/\/[^\s<>")]+)/g, '<a href="$1" target="_blank" rel="noreferrer">$1</a>')
  s = s.replace(/(?:^|\n)((?:[-*] .*(?:\n|$))+)/g, (_m, b: string) => "\n<ul>" + b.trim().split("\n").map(l => "<li>" + l.replace(/^[-*] /, "") + "</li>").join("") + "</ul>")
  s = s.replace(/(?:^|\n)((?:\d+\. .*(?:\n|$))+)/g, (_m, b: string) => "\n<ol>" + b.trim().split("\n").map(l => "<li>" + l.replace(/^\d+\. /, "") + "</li>").join("") + "</ol>")
  return s.split(/\n{2,}/).map(p => /^\s*<(h\d|ul|ol|pre)/.test(p) ? p : "<p>" + p.replace(/\n/g, "<br>") + "</p>").join("")
}

export default function App() {
  const [threads, setThreads] = useState<Thread[]>([])
  const [cur, setCur] = useState<string>(newId())
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [title, setTitle] = useState("Новый чат")
  const [mode, setMode] = useState("")
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const [online, setOnline] = useState(false)
  const [sideOpen, setSideOpen] = useState(true)
  const [attached, setAttached] = useState<string[]>([])
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)
  const ta = useRef<HTMLTextAreaElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const recRef = useRef<MediaRecorder | null>(null)

  const loadThreads = useCallback(async () => {
    try { const r = await fetch(`/chats?user_id=${uid}`); setThreads(await r.json()); setOnline(true) }
    catch { setOnline(false) }
  }, [])
  useEffect(() => { loadThreads(); const t = setInterval(loadThreads, 15000); return () => clearInterval(t) }, [loadThreads])
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }) }, [msgs])

  const newChat = () => { setCur(newId()); setMsgs([]); setTitle("Новый чат"); setMode(""); setAttached([]); ta.current?.focus() }
  const openThread = async (id: string) => {
    setCur(id); setAttached([])
    try { const d = await (await fetch(`/chats/${id}`)).json(); setTitle(d.thread?.title || "Чат"); setMsgs(d.messages || []) }
    catch { setMsgs([]) }
  }
  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); await fetch(`/chats/${id}`, { method: "DELETE" })
    if (id === cur) newChat(); else loadThreads()
  }

  const sendText = async (text: string) => {
    text = text.trim(); if (!text || busy) return
    setBusy(true); setInput(""); if (ta.current) ta.current.style.height = "auto"
    setMsgs(m => [...m, { role: "user", content: text }, { role: "assistant", content: "…" }])
    setAttached([])
    try {
      const r = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ user_id: uid, thread_id: cur, query: text }) })
      if (!r.ok) throw new Error("HTTP " + r.status)
      const d = await r.json()
      if (d.mode) setMode(d.mode); if (d.title) setTitle(d.title)
      // мягкий «стрим»: ответ набегает, markdown формируется на лету (из мутного в яркое)
      const full = d.answer || "(пусто)"
      const total = full.length, per = Math.max(1, Math.ceil(total / Math.min(total, 90)))
      await new Promise<void>(res => {
        let n = 0
        const tick = () => {
          n = Math.min(total, n + per)
          setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: "assistant", content: full.slice(0, n) }; return c })
          if (n < total) window.setTimeout(tick, 16); else res()
        }
        tick()
      })
      loadThreads()
    } catch (e: any) {
      const net = /load failed|failed to fetch|networkerror/i.test(String(e?.message || e))
      const note = net
        ? "связь с агентом прервалась (сервер занят долгой задачей или перезапустился). Текст сохранён — нажми ↑, чтобы повторить."
        : String(e?.message || e)
      setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: "assistant", content: "⚠ " + note }; return c })
      setInput(text)  // вернуть текст для повтора
    } finally { setBusy(false); ta.current?.focus() }
  }
  const send = () => sendText(input)
  const onKey = (e: React.KeyboardEvent) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() } }
  const grow = (el: HTMLTextAreaElement) => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 200) + "px" }

  const onFiles = async (files: FileList | null) => {
    if (!files) return
    for (const f of Array.from(files)) {
      const fd = new FormData(); fd.append("file", f)
      try { await fetch(`/upload?thread_id=${cur}`, { method: "POST", body: fd }); setAttached(a => [...a, f.name]) } catch { /* ignore */ }
    }
  }
  const startRec = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream); const chunks: Blob[] = []
      mr.ondataavailable = e => chunks.push(e.data)
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const fd = new FormData(); fd.append("file", new Blob(chunks, { type: "audio/webm" }), "rec.webm")
        setTranscribing(true)
        try { const d = await (await fetch("/transcribe", { method: "POST", body: fd })).json(); if (d.text) setInput(v => (v ? v + " " : "") + d.text) }
        finally { setTranscribing(false); setTimeout(() => ta.current && grow(ta.current), 0) }
      }
      mr.start(); recRef.current = mr; setRecording(true)
    } catch { alert("Нет доступа к микрофону") }
  }
  const stopRec = () => { recRef.current?.stop(); setRecording(false) }

  const prompts = ["Сравни 3 ноутбука до 100к и посоветуй", "Объясни attention в трансформере со ссылками", "Посчитай сложный процент по вкладу"]
  const iconBtn = "grid place-items-center w-8 h-8 rounded-lg transition-colors shrink-0"

  return (
    <div className="flex h-full" style={{ background: "var(--bg)" }}>
      {/* Sidebar */}
      <motion.aside initial={false} animate={{ width: sideOpen ? 256 : 0 }} transition={{ duration: 0.28, ease: [0.22, 0.7, 0.2, 1] }}
        className="relative z-10 shrink-0 overflow-hidden" style={{ background: "var(--surface)", boxShadow: "var(--sh2)" }}>
        <div className="w-64 h-full flex flex-col">
          <div className="flex items-center gap-2.5 px-5 pt-5 pb-4">
            <div className="grid place-items-center w-7 h-7 rounded-[9px] text-[13px] font-bold"
              style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: "var(--sh1)" }}>S</div>
            <span className="text-[14px] font-bold tracking-tight" style={{ color: "var(--ink)" }}>self&#8209;extension</span>
          </div>
          <div className="px-3 pb-3">
            <motion.button whileTap={{ scale: 0.97 }} onClick={newChat}
              className="w-full flex items-center gap-2 px-3.5 py-2.5 text-[13px] font-semibold rounded-xl transition-colors"
              style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: "var(--sh1)" }}
              onMouseEnter={e => (e.currentTarget.style.background = "var(--accent-h)")}
              onMouseLeave={e => (e.currentTarget.style.background = "var(--accent)")}>
              <Plus size={16} strokeWidth={2.5} /> Новый чат
            </motion.button>
          </div>
          <div className="px-5 pb-2 text-[11px] font-semibold tracking-wide" style={{ color: "var(--faint)" }}>История</div>
          <div className="flex-1 overflow-y-auto px-2.5 pb-3 min-h-0">
            {threads.length === 0 && <div className="px-2.5 py-2 text-[12.5px]" style={{ color: "var(--faint)" }}>пока пусто</div>}
            {threads.map(t => {
              const active = t.thread_id === cur
              return (
                <motion.div key={t.thread_id} whileTap={{ scale: 0.98 }} onClick={() => openThread(t.thread_id)}
                  className="group relative flex items-center px-3 py-2 my-0.5 rounded-xl cursor-pointer text-[13px] truncate transition-all"
                  style={{ background: active ? "var(--surface2)" : "transparent", color: active ? "var(--ink)" : "var(--muted)", boxShadow: active ? "var(--sh1)" : "none", fontWeight: active ? 600 : 400 }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.background = "var(--bg)" }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.background = "transparent" }}>
                  {!!t.favorite && <Star size={11} className="mr-1.5 shrink-0" fill="var(--accent)" stroke="var(--accent)" />}
                  <span className="truncate">{t.title || "без названия"}</span>
                  <button onClick={e => del(t.thread_id, e)} className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 rounded-md transition-opacity" style={{ color: "var(--faint)" }}><X size={12} /></button>
                </motion.div>
              )
            })}
          </div>
          <div className="flex items-center gap-2 px-4 py-3" style={{ color: "var(--faint)" }}>
            <button onClick={() => setShowSettings(true)} className={iconBtn}
              onMouseEnter={e => (e.currentTarget.style.background = "var(--bg)")} onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
              title="Настройки"><Settings size={16} /></button>
            <span className="flex items-center gap-1.5 text-[11px] font-mono truncate">
              <span className="w-[7px] h-[7px] rounded-full" style={{ background: online ? "#7e8a4e" : "var(--faint)" }} />
              {online ? uid : "offline"}
            </span>
          </div>
        </div>
      </motion.aside>

      {/* Main */}
      <main className="flex flex-col flex-1 min-w-0 min-h-0">
        <div className="flex items-center gap-2 px-6 pt-4 pb-2">
          <button onClick={() => setSideOpen(s => !s)} className={iconBtn} style={{ color: "var(--muted)" }}
            onMouseEnter={e => (e.currentTarget.style.background = "var(--surface)")} onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
            title="Свернуть панель"><PanelLeft size={17} /></button>
          <span className="text-[14px] font-semibold truncate" style={{ color: "var(--ink)" }}>{title}</span>
          {mode && <span className="ml-auto text-[11px] font-medium px-2 py-0.5 rounded-md" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>{mode}</span>}
        </div>

        <div ref={scroller} className="flex-1 overflow-y-auto min-h-0">
          {msgs.length === 0 ? (
            <div className="h-full flex flex-col justify-center max-w-3xl mx-auto px-7 w-full pb-10">
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
                <div className="grid place-items-center w-12 h-12 rounded-2xl text-[22px] font-bold mb-6"
                  style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: "var(--sh2)" }}>S</div>
                <h1 className="text-[34px] font-extrabold tracking-[-0.035em] leading-[1.05]" style={{ color: "var(--ink)" }}>С чего начнём?</h1>
                <p className="mt-3 text-[15px]" style={{ color: "var(--muted)" }}>Поиск, анализ, вычисления, браузер — спокойно и по делу. И я запомню тебя.</p>
                <div className="mt-7 grid gap-2.5 max-w-[480px]">
                  {prompts.map((p, i) => (
                    <motion.button key={i} whileHover={{ y: -2 }} whileTap={{ scale: 0.99 }} onClick={() => sendText(p)}
                      className="text-left px-4 py-3.5 text-[13.5px] rounded-lg transition-shadow"
                      style={{ background: "var(--surface2)", color: "var(--ink)", boxShadow: "var(--sh1)" }}
                      onMouseEnter={e => (e.currentTarget.style.boxShadow = "var(--sh2)")}
                      onMouseLeave={e => (e.currentTarget.style.boxShadow = "var(--sh1)")}>{p}</motion.button>
                  ))}
                </div>
              </motion.div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-7 py-8">
              <AnimatePresence initial={false}>
                {msgs.map((m, i) => (
                  m.role === "user" ? (
                    <motion.div key={i} initial={{ opacity: 0, y: 6, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.4, ease: [0.22, 0.7, 0.2, 1] }} className="pt-3 pb-6">
                      <div className="text-[20px] font-semibold leading-snug tracking-[-0.02em] whitespace-pre-wrap" style={{ color: "var(--ink)" }}>{m.content}</div>
                    </motion.div>
                  ) : (
                    <motion.div key={i} initial={{ opacity: 0.3, y: 6, filter: "blur(7px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.55, ease: [0.22, 0.7, 0.2, 1] }} className="pb-10">
                      <div className="flex items-center gap-2 mb-3.5">
                        <span className="grid place-items-center w-[18px] h-[18px] rounded-[5px] text-[9px] font-bold" style={{ background: "var(--accent)", color: "var(--accent-fg)" }}>S</span>
                        <span className="text-[11px] font-bold tracking-[0.1em]" style={{ color: "var(--accent)" }}>АГЕНТ</span>
                      </div>
                      {m.content === "…" || m.content === ""
                        ? <Thinking />
                        : <div className="md" style={{ color: "var(--ink)" }} dangerouslySetInnerHTML={{ __html: md(m.content) }} />}
                    </motion.div>
                  )
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>

        <div className="px-7 pb-7 pt-2">
          <div className="max-w-3xl mx-auto">
            {attached.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-2 px-1">
                {attached.map((n, i) => (
                  <span key={i} className="flex items-center gap-1.5 text-[12px] px-2.5 py-1 rounded-lg" style={{ background: "var(--accent-soft)", color: "var(--ink)" }}>
                    <FileText size={12} /> {n}
                  </span>
                ))}
              </div>
            )}
            <div className="flex items-end gap-2 rounded-xl px-3 py-2.5" style={{ background: "var(--surface2)", boxShadow: "var(--sh3)" }}>
              <input ref={fileInput} type="file" multiple className="hidden" onChange={e => { onFiles(e.target.files); e.target.value = "" }} />
              <button onClick={() => fileInput.current?.click()} className={iconBtn} style={{ color: "var(--muted)" }}
                onMouseEnter={e => (e.currentTarget.style.background = "var(--surface)")} onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                title="Прикрепить файл"><Paperclip size={18} /></button>
              <textarea ref={ta} rows={1} value={input} placeholder={transcribing ? "распознаю речь…" : "Сообщение агенту…"}
                onChange={e => { setInput(e.target.value); grow(e.target) }} onKeyDown={onKey}
                className="flex-1 resize-none bg-transparent outline-none text-[14.5px] leading-relaxed py-1.5 max-h-[200px]" style={{ color: "var(--ink)" }} />
              <button onClick={recording ? stopRec : startRec} className={iconBtn}
                style={{ color: recording ? "#fff" : "var(--muted)", background: recording ? "var(--accent)" : "transparent" }}
                onMouseEnter={e => { if (!recording) e.currentTarget.style.background = "var(--surface)" }}
                onMouseLeave={e => { if (!recording) e.currentTarget.style.background = "transparent" }}
                title={recording ? "Остановить запись" : "Записать голос"}>{recording ? <Square size={15} fill="#fff" /> : <Mic size={18} />}</button>
              <motion.button whileTap={{ scale: 0.9 }} onClick={send} disabled={busy || !input.trim()}
                className="grid place-items-center w-9 h-9 rounded-lg shrink-0 transition-all disabled:opacity-35"
                style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: input.trim() ? "var(--sh1)" : "none" }}><ArrowUp size={17} strokeWidth={2.5} /></motion.button>
            </div>
          </div>
        </div>
      </main>

      <AnimatePresence>{showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}</AnimatePresence>
    </div>
  )
}

function SettingsModal({ onClose }: { onClose: () => void }) {
  const [cfg, setCfg] = useState<Cfg>({})
  const [provider, setProvider] = useState("openrouter")
  const [baseUrl, setBaseUrl] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [model, setModel] = useState("")
  const [codeModel, setCodeModel] = useState("")
  const [deepModel, setDeepModel] = useState("")
  const [workMode, setWorkMode] = useState("auto-accept")
  const [forceMode, setForceMode] = useState("")
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState("")
  useEffect(() => {
    (async () => {
      try {
        const c = await (await fetch("/settings")).json()
        setCfg(c); setProvider(c.provider || "openrouter"); setBaseUrl(c.base_url || "")
        setModel(c.model || ""); setCodeModel(c.code_model || ""); setDeepModel(c.deep_model || "")
        setWorkMode(c.work_mode || "auto-accept"); setForceMode(c.force_mode || "")
      } catch { /* */ }
    })()
  }, [])
  const save = async () => {
    setSaving(true); setMsg("")
    try {
      const body: any = { provider, base_url: baseUrl, model, code_model: codeModel, deep_model: deepModel, work_mode: workMode, force_mode: forceMode }
      if (apiKey) body.api_key = apiKey
      const d = await (await fetch("/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json()
      setMsg(d.message || ""); setCfg(c => ({ ...c, active: d.active, api_key_source: d.api_key_source })); setApiKey("")
    } catch (e: any) { setMsg("ошибка: " + e.message) } finally { setSaving(false) }
  }
  const field = "w-full px-3 py-2.5 rounded-lg text-[13.5px] outline-none font-mono"
  const fst = { background: "var(--surface)", color: "var(--ink)" } as React.CSSProperties
  const Lbl = ({ children }: { children: any }) => <label className="block text-[11.5px] font-semibold mb-1.5" style={{ color: "var(--muted)" }}>{children}</label>
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}
      className="fixed inset-0 z-50 grid place-items-center p-5" style={{ background: "rgba(9,30,66,.54)" }}>
      <motion.div initial={{ scale: 0.97, y: 10 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.97, opacity: 0 }} transition={{ duration: 0.18 }}
        onClick={e => e.stopPropagation()} className="w-full max-w-[460px] max-h-[88vh] flex flex-col rounded-2xl" style={{ background: "var(--surface)", boxShadow: "var(--sh3)" }}>
        <div className="flex items-center px-6 pt-5 pb-4">
          <h2 className="text-[18px] font-bold tracking-tight" style={{ color: "var(--ink)" }}>Настройки</h2>
          <button onClick={onClose} className="ml-auto grid place-items-center w-8 h-8 rounded-lg transition-colors" style={{ color: "var(--muted)" }}
            onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")} onMouseLeave={e => (e.currentTarget.style.background = "transparent")}><X size={18} /></button>
        </div>
        <div className="px-6 pb-4 overflow-y-auto space-y-4">
          <div><Lbl>Провайдер</Lbl>
            <Dropdown value={provider} onChange={setProvider} options={[{ v: "openrouter", l: "OpenRouter" }, { v: "ollama", l: "Ollama (локально)" }]} /></div>
          <div className="grid grid-cols-2 gap-3">
            <div><Lbl>Основная модель</Lbl><input value={model} onChange={e => setModel(e.target.value)} placeholder="google/gemini-2.5-flash-lite" className={field} style={fst} /></div>
            <div><Lbl>Модель кода</Lbl><input value={codeModel} onChange={e => setCodeModel(e.target.value)} placeholder="deepseek/deepseek-v4-flash" className={field} style={fst} /></div>
          </div>
          <div><Lbl>Модель тяжёлого ревью (deep)</Lbl><input value={deepModel} onChange={e => setDeepModel(e.target.value)} placeholder="модель для heavy-режима" className={field} style={fst} /></div>
          <div><Lbl>Endpoint (base_url)</Lbl><input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="https://openrouter.ai/api/v1" className={field} style={fst} /></div>
          <div><Lbl>API-ключ · сейчас: {cfg.api_key_source || "—"}</Lbl>
            <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="вставь ключ (необязательно)" className={field} style={fst} /></div>
          <div className="grid grid-cols-2 gap-3">
            <div><Lbl>Режим работы</Lbl>
              <Dropdown value={workMode} onChange={setWorkMode} options={[
                { v: "manual", l: "Спрашивать" }, { v: "auto-accept", l: "Без спроса" }, { v: "auto", l: "Автономия" }]} /></div>
            <div><Lbl>Режим мышления</Lbl>
              <Dropdown value={forceMode} onChange={setForceMode} options={[
                { v: "", l: "Авто" }, { v: "fast", l: "Быстрый" }, { v: "reason", l: "Рассуждение" },
                { v: "act", l: "Действие" }, { v: "deliberate", l: "Глубокий" }, { v: "heavy", l: "Тяжёлый" }]} /></div>
          </div>
          {cfg.active && <div className="text-[11.5px] leading-relaxed" style={{ color: "var(--faint)" }}>Активно: {cfg.active}</div>}
          {msg && <div className="text-[12.5px] font-medium" style={{ color: "var(--accent)" }}>{msg}</div>}
        </div>
        <div className="px-6 py-4">
          <motion.button whileTap={{ scale: 0.98 }} onClick={save} disabled={saving}
            className="w-full py-2.5 rounded-lg text-[13.5px] font-semibold disabled:opacity-50"
            style={{ background: "var(--accent)", color: "var(--accent-fg)" }}>{saving ? "проверяю…" : "Сохранить и проверить"}</motion.button>
        </div>
      </motion.div>
    </motion.div>
  )
}

function Dropdown({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { v: string; l: string }[] }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener("mousedown", h); return () => document.removeEventListener("mousedown", h)
  }, [])
  const cur = options.find(o => o.v === value)
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center px-3 py-2.5 rounded-lg text-[13.5px] transition-colors"
        style={{ background: "var(--surface)", color: "var(--ink)" }}>
        {cur?.l || value}
        <ChevronDown size={16} className="ml-auto transition-transform" style={{ color: "var(--muted)", transform: open ? "rotate(180deg)" : "none" }} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -6, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -6, scale: 0.98 }} transition={{ duration: 0.14 }}
            className="absolute left-0 right-0 mt-1.5 p-1 rounded-lg z-10" style={{ background: "var(--surface2)", boxShadow: "var(--sh3)" }}>
            {options.map(o => (
              <button key={o.v} onClick={() => { onChange(o.v); setOpen(false) }}
                className="w-full flex items-center px-2.5 py-2 rounded-md text-[13.5px] transition-colors"
                style={{ color: "var(--ink)", background: o.v === value ? "var(--accent-soft)" : "transparent" }}
                onMouseEnter={e => { if (o.v !== value) e.currentTarget.style.background = "var(--surface)" }}
                onMouseLeave={e => { if (o.v !== value) e.currentTarget.style.background = "transparent" }}>
                {o.l}{o.v === value && <Check size={15} className="ml-auto" style={{ color: "var(--accent)" }} />}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

const THINK_WORDS = ["Думаю", "Вникаю", "Ищу", "Считаю", "Соображаю", "Кручу шестерёнки", "Готовлю", "Собираю"]
function Thinking() {
  const [i, setI] = useState(0)
  useEffect(() => { const t = setInterval(() => setI(v => (v + 1) % THINK_WORDS.length), 2800); return () => clearInterval(t) }, [])
  return (
    <div className="relative h-[24px] py-1" style={{ color: "var(--muted)" }}>
      <AnimatePresence>
        <motion.span key={i} className="absolute left-0 top-1 text-[15px] font-medium whitespace-nowrap"
          initial={{ opacity: 0, filter: "blur(7px)" }} animate={{ opacity: 1, filter: "blur(0px)" }}
          exit={{ opacity: 0, filter: "blur(7px)" }} transition={{ duration: 0.75, ease: "easeInOut" }}>
          {THINK_WORDS[i]}…
        </motion.span>
      </AnimatePresence>
    </div>
  )
}
