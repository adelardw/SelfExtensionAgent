import { useEffect, useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { marked } from "marked"
import DOMPurify from "dompurify"
import { Plus, ArrowUp, Star, X, PanelLeft, Settings, Paperclip, Mic, Square, ChevronDown, Check } from "lucide-react"

marked.setOptions({ gfm: true, breaks: true })

type Thread = { thread_id: string; title: string; favorite?: number }
type Msg = { role: "user" | "assistant"; content: string }
type Cfg = { provider?: string; base_url?: string; api_key_source?: string; active?: string;
  model?: string; code_model?: string; deep_model?: string; work_mode?: string; force_mode?: string;
  searxng_url?: string; bridge_connected?: boolean; bridge_token?: string; bridge_port?: number }

// СТАБИЛЬНЫЙ user_id (single-owner). Раньше был случайный gui-XXXX в localStorage, но в pywebview
// порт сервера случаен каждый запуск → origin меняется → localStorage сбрасывается → новый uid →
// история «пропадала». Фикс — фиксированный "local" (история накапливается в БД и видна при открытии).
const uid = (() => {
  let u = localStorage.getItem("agent_uid")
  if (!u || u.startsWith("gui-")) { u = "local"; try { localStorage.setItem("agent_uid", u) } catch { /* ephemeral webview */ } }
  return u
})()
const newId = () => (crypto.randomUUID ? crypto.randomUUID() : "t-" + Date.now())
const esc = (s: string) => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!))
const escA = (s: string) => esc(s).replace(/"/g, "&quot;")  // безопасно для значения атрибута

// Блок ```sea-gallery``` (строки: img_url ||| подпись ||| источник) → окно-галерея.
// Картинки из веб-поиска = внешний контент → жёстко экранируем + только http(s); финал ещё и
// прогоняется через DOMPurify в md(). Комментарий агента сверху/снизу — обычный markdown вокруг.
function renderGallery(body: string): string {
  let title = ""
  const cells: string[] = []
  for (const raw of body.split("\n")) {
    const l = raw.trim()
    if (!l) continue
    if (l.startsWith("#")) { title = l.replace(/^#+\s*/, ""); continue }
    const [img, cap, src] = l.split("|||").map(x => (x || "").trim())
    if (!/^https?:\/\//i.test(img)) continue
    const figcap = cap ? `<figcaption>${esc(cap)}</figcaption>` : ""
    const inner = `<img src="${escA(img)}" alt="${escA(cap || title)}" loading="lazy">${figcap}`
    cells.push(/^https?:\/\//i.test(src)
      ? `<a class="sea-cell" href="${escA(src)}" target="_blank" rel="noopener noreferrer">${inner}</a>`
      : `<figure class="sea-cell">${inner}</figure>`)
  }
  if (!cells.length) return ""
  const head = `<div class="sea-gal-head">🖼 ${esc(title) || "Images"} · ${cells.length}</div>`
  return `<div class="sea-gallery">${head}<div class="sea-gal-grid">${cells.join("")}</div></div>`
}

function md(s: string): string {
  const gals: string[] = []
  // вынуть галереи ДО marked, чтобы он их не трогал; вернуть после — и всё санитизируем DOMPurify
  const pre = (s || "").replace(/```sea-gallery\n([\s\S]*?)\n```/g, (_m, b) => {
    gals.push(renderGallery(b)); return `@@GAL${gals.length - 1}@@`
  })
  try {
    let html = marked.parse(pre, { async: false }) as string
    html = html.replace(/<p>\s*@@GAL(\d+)@@\s*<\/p>/g, (_m, i) => gals[+i] || "")
               .replace(/@@GAL(\d+)@@/g, (_m, i) => gals[+i] || "")
    return DOMPurify.sanitize(html, { ADD_ATTR: ["target", "rel", "loading"] })
  } catch {
    return esc(s).replace(/\n/g, "<br>")
  }
}

export default function App() {
  const [threads, setThreads] = useState<Thread[]>([])
  const [cur, setCur] = useState<string>(newId())
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [title, setTitle] = useState("New chat")
  const [mode, setMode] = useState("")
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const [online, setOnline] = useState(false)
  const [sideOpen, setSideOpen] = useState(true)
  const [attached, setAttached] = useState<string[]>([])
  const [uploading, setUploading] = useState(0)   // файлов в процессе загрузки/обработки (PDF парсится)
  const [attachErr, setAttachErr] = useState("")  // ошибка прикрепления (показываем, не глотаем)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [pending, setPending] = useState<{ run_id: string; questions?: { question: string; options: string[]; why?: string }[]; confirm?: string } | null>(null)
  const [progress, setProgress] = useState("")
  const [steps, setSteps] = useState<string[]>([])
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

  const newChat = () => { setCur(newId()); setMsgs([]); setTitle("New chat"); setMode(""); setAttached([]); setPending(null); setSteps([]); setProgress(""); ta.current?.focus() }
  const openThread = async (id: string) => {
    setCur(id); setAttached([]); setPending(null); setSteps([]); setProgress("")
    try { const d = await (await fetch(`/chats/${id}`)).json(); setTitle(d.thread?.title || "Chat"); setMsgs(d.messages || []) }
    catch { setMsgs([]) }
  }
  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); await fetch(`/chats/${id}`, { method: "DELETE" })
    if (id === cur) newChat(); else loadThreads()
  }

  const setLastAssistant = (content: string) =>
    setMsgs(m => { const c = [...m]; if (c.length) c[c.length - 1] = { role: "assistant", content }; return c })
  const streamAnswer = (full: string) => new Promise<void>(res => {
    const total = full.length, per = Math.max(1, Math.ceil(total / Math.min(total || 1, 90)))
    let n = 0
    const tick = () => { n = Math.min(total, n + per); setLastAssistant(full.slice(0, n)); n < total ? window.setTimeout(tick, 16) : res() }
    tick()
  })
  const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

  const poll = async (run_id: string): Promise<void> => {
    for (; ;) {
      await sleep(650)
      let d: any
      try { d = await (await fetch(`/run/${run_id}`)).json() } catch { continue }  // транзиент → ещё раз
      if (d.steps) setSteps(d.steps)
      if (d.status === "running") { setProgress(d.progress || ""); continue }
      if (d.status === "waiting") {
        if (d.pending?.type === "clarify") { setPending({ run_id, questions: d.pending.questions || [] }); return }
        if (d.pending?.type === "confirm") { setPending({ run_id, confirm: d.pending.text }); return }
        setProgress(d.progress || ""); continue
      }
      if (d.status === "done") { setProgress(""); await streamAnswer(d.answer || "(empty)"); if (d.mode) setMode(d.mode); if (d.title) setTitle(d.title); loadThreads(); setBusy(false); return }
      if (d.status === "error") { setProgress(""); setLastAssistant("⚠ error: " + (d.error || "")); setBusy(false); return }
      setProgress(""); setLastAssistant("⚠ run not found (server restarted?)"); setBusy(false); return
    }
  }

  const sendText = async (text: string) => {
    text = text.trim(); if (!text || busy) return
    setBusy(true); setInput(""); setSteps([]); setProgress(""); if (ta.current) ta.current.style.height = "auto"
    setMsgs(m => [...m, { role: "user", content: text }, { role: "assistant", content: "…" }])
    setAttached([])
    try {
      const r = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ user_id: uid, thread_id: cur, query: text }) })
      if (!r.ok) throw new Error("HTTP " + r.status)
      const { run_id } = await r.json()
      await poll(run_id)
    } catch (e: any) {
      const net = /load failed|failed to fetch|networkerror/i.test(String(e?.message || e))
      setLastAssistant("⚠ " + (net ? "connection lost — your text is saved, press ↑ to retry" : String(e?.message || e)))
      setInput(text); setBusy(false)
    }
    ta.current?.focus()
  }

  const submitClarify = async (answers: string[]) => {
    if (!pending) return
    const run_id = pending.run_id; setPending(null); setLastAssistant("…")
    try { await fetch(`/run/${run_id}/respond`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ answers }) }) } catch { /* */ }
    poll(run_id)
  }
  const submitConfirm = async (value: string) => {
    if (!pending) return
    const run_id = pending.run_id; setPending(null); setLastAssistant("…")
    try { await fetch(`/run/${run_id}/respond`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ value }) }) } catch { /* */ }
    poll(run_id)
  }
  const send = () => sendText(input)
  const onKey = (e: React.KeyboardEvent) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() } }
  const grow = (el: HTMLTextAreaElement) => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 200) + "px" }

  const onFiles = async (files: FileList | null) => {
    if (!files) return
    const list = Array.from(files)
    setUploading(u => u + list.length)
    for (const f of list) {
      const fd = new FormData(); fd.append("file", f)
      try { await fetch(`/upload?thread_id=${cur}`, { method: "POST", body: fd }); setAttached(a => [...a, f.name]) }
      catch { /* ignore */ }
      finally { setUploading(u => Math.max(0, u - 1)) }
    }
  }
  // Прикрепить файл: в десктопе (pywebview) WKWebView не открывает диалог по программному клику —
  // зовём нативный pywebview.api.pick_files() и отдаём ЛОКАЛЬНЫЕ пути серверу; в браузере — <input>.
  const attachLocal = async (paths: string[]) => {
    if (!paths.length) return
    setUploading(u => u + paths.length); setAttachErr("")   // PDF парсится несколько сек → спиннер
    try {
      const d = await (await fetch("/attach_local", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: cur, paths }),
      })).json()
      if (d.names?.length) setAttached(a => [...a, ...d.names])
      if (d.errors?.length) setAttachErr(`couldn't attach: ${d.errors.join(", ")}`)
    } catch { setAttachErr("file upload error") }
    finally { setUploading(u => Math.max(0, u - paths.length)) }
  }
  const detach = async (name: string) => {
    setAttached(a => a.filter(n => n !== name))   // мгновенно убрать из UI
    try { await fetch("/detach", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ thread_id: cur, name }) }) } catch { /* */ }
  }
  const pickFiles = async () => {
    const pw = (window as any).pywebview
    if (pw?.api?.pick_files) {
      try { await attachLocal(await pw.api.pick_files() || []) } catch { /* ignore */ }
    } else {
      fileInput.current?.click()
    }
  }
  const startRec = async () => {
    // Десктоп (pywebview): WKWebView не умеет getUserMedia → пишем на СЕРВЕРЕ (ffmpeg).
    if ((window as any).pywebview) {
      try {
        const r = await (await fetch(`/voice/start?thread_id=${cur}`, { method: "POST" })).json()
        if (r.ok) { recRef.current = "server" as any; setRecording(true) }
        else alert(r.error || "No microphone access")
      } catch { alert("No microphone access") }
      return
    }
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
    } catch { alert("No microphone access") }
  }
  const stopRec = async () => {
    if (recRef.current === ("server" as any)) {   // десктоп: стоп серверной записи + расшифровка
      recRef.current = null; setRecording(false); setTranscribing(true)
      try {
        const d = await (await fetch(`/voice/stop?thread_id=${cur}`, { method: "POST" })).json()
        if (d.text) setInput(v => (v ? v + " " : "") + d.text)
      } finally { setTranscribing(false); setTimeout(() => ta.current && grow(ta.current), 0) }
      return
    }
    recRef.current?.stop(); setRecording(false)
  }

  const prompts = ["Compare 3 laptops under $1000 and recommend one", "Explain attention in transformers with references", "Compute compound interest on a deposit"]
  const iconBtn = "grid place-items-center w-8 h-8 rounded-lg transition-colors shrink-0"

  return (
    <div className="flex h-full" style={{ background: "var(--bg)" }}>
      {/* Sidebar */}
      <motion.aside initial={false} animate={{ width: sideOpen ? 256 : 0 }} transition={{ duration: 0.28, ease: [0.22, 0.7, 0.2, 1] }}
        className="relative z-10 shrink-0 overflow-hidden" style={{ background: "var(--surface)", boxShadow: "var(--sh2)" }}>
        <div className="w-64 h-full flex flex-col">
          <div className="flex items-center gap-2.5 px-5 pt-5 pb-4">
            <div className="grid place-items-center w-7 h-7 rounded-[9px] text-[12px] font-extrabold tracking-tight"
              style={{ background: "var(--sea-grad)", color: "var(--accent-fg)", boxShadow: "var(--sh1)" }}>🌊</div>
            <span className="sea-shimmer text-[15px] font-extrabold tracking-tight">SEA</span>
            <span className="text-[12px] font-medium tracking-tight" style={{ color: "var(--faint)" }}>self&#8209;extension</span>
          </div>
          <div className="px-3 pb-3">
            <motion.button whileTap={{ scale: 0.97 }} onClick={newChat}
              className="w-full flex items-center gap-2 px-3.5 py-2.5 text-[13px] font-semibold rounded-xl transition-colors"
              style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: "var(--sh1)" }}
              onMouseEnter={e => (e.currentTarget.style.background = "var(--accent-h)")}
              onMouseLeave={e => (e.currentTarget.style.background = "var(--accent)")}>
              <Plus size={16} strokeWidth={2.5} /> New chat
            </motion.button>
          </div>
          <div className="px-5 pb-2 text-[11px] font-semibold tracking-wide" style={{ color: "var(--faint)" }}>History</div>
          <div className="flex-1 overflow-y-auto px-2.5 pb-3 min-h-0">
            {threads.length === 0 && <div className="px-2.5 py-2 text-[12.5px]" style={{ color: "var(--faint)" }}>empty</div>}
            {threads.map(t => {
              const active = t.thread_id === cur
              return (
                <motion.div key={t.thread_id} whileTap={{ scale: 0.98 }} onClick={() => openThread(t.thread_id)}
                  className="group relative flex items-center px-3 py-2 my-0.5 rounded-xl cursor-pointer text-[13px] truncate transition-all"
                  style={{ background: active ? "var(--surface2)" : "transparent", color: active ? "var(--ink)" : "var(--muted)", boxShadow: active ? "var(--sh1)" : "none", fontWeight: active ? 600 : 400 }}
                  onMouseEnter={e => { if (!active) e.currentTarget.style.background = "var(--bg)" }}
                  onMouseLeave={e => { if (!active) e.currentTarget.style.background = "transparent" }}>
                  {!!t.favorite && <Star size={11} className="mr-1.5 shrink-0" fill="var(--accent)" stroke="var(--accent)" />}
                  <span className="truncate">{t.title || "untitled"}</span>
                  <button onClick={e => del(t.thread_id, e)} className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 rounded-md transition-opacity" style={{ color: "var(--faint)" }}><X size={12} /></button>
                </motion.div>
              )
            })}
          </div>
          <div className="flex items-center gap-2 px-4 py-3" style={{ color: "var(--faint)" }}>
            <button onClick={() => setShowSettings(true)} className={iconBtn}
              onMouseEnter={e => (e.currentTarget.style.background = "var(--bg)")} onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
              title="Settings"><Settings size={16} /></button>
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
            title="Toggle sidebar"><PanelLeft size={17} /></button>
          <span className="text-[14px] font-semibold truncate" style={{ color: "var(--ink)" }}>{title}</span>
          {mode && <span className="ml-auto text-[11px] font-medium px-2 py-0.5 rounded-md" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>{mode}</span>}
        </div>

        <div ref={scroller} className="flex-1 overflow-y-auto min-h-0">
          {msgs.length === 0 ? (
            <div className="h-full flex flex-col justify-center max-w-3xl mx-auto px-7 w-full pb-10">
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
                <div className="grid place-items-center w-12 h-12 rounded-2xl text-[22px] font-bold mb-6"
                  style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: "var(--sh2)" }}>S</div>
                <h1 className="text-[34px] font-extrabold tracking-[-0.035em] leading-[1.05]" style={{ color: "var(--ink)" }}>Where do we start?</h1>
                <p className="mt-3 text-[15px]" style={{ color: "var(--muted)" }}>Search, analysis, computation, browser — calm and to the point. And I'll remember you.</p>
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
                        <span className="text-[11px] font-bold tracking-[0.1em]" style={{ color: "var(--accent)" }}>AGENT</span>
                      </div>
                      {m.content === "…" || m.content === ""
                        ? <Working steps={steps} current={progress} />
                        : <div className="md" style={{ color: "var(--ink)" }} dangerouslySetInnerHTML={{ __html: md(m.content) }} />}
                    </motion.div>
                  )
                ))}
              </AnimatePresence>
              {pending?.questions && <ClarifyCard questions={pending.questions} onSubmit={submitClarify} />}
              {pending?.confirm && <ConfirmCard text={pending.confirm} onConfirm={submitConfirm} />}
            </div>
          )}
        </div>

        <div className="px-7 pb-7 pt-2">
          <div className="max-w-3xl mx-auto">
            {(attached.length > 0 || uploading > 0 || attachErr) && (
              <div className="flex flex-wrap gap-2 mb-2 px-1">
                {attached.map((n, i) => (
                  <span key={i} className="group flex items-center gap-1.5 text-[12px] pl-2.5 pr-1.5 py-1 rounded-lg" style={{ background: "var(--accent-soft)", color: "var(--ink)" }}>
                    <Check size={12} style={{ color: "var(--accent)" }} /> {n}
                    <button onClick={() => detach(n)} title="Remove attachment"
                      className="grid place-items-center w-4 h-4 rounded opacity-45 hover:opacity-100 transition-opacity" style={{ color: "var(--muted)" }}>
                      <X size={12} />
                    </button>
                  </span>
                ))}
                {uploading > 0 && (
                  <span className="flex items-center gap-1.5 text-[12px] px-2.5 py-1 rounded-lg" style={{ background: "var(--accent-soft)", color: "var(--muted)" }}>
                    <span className="inline-block w-3 h-3 rounded-full border-2 animate-spin" style={{ borderColor: "var(--accent) transparent transparent transparent" }} />
                    Processing{uploading > 1 ? ` ×${uploading}` : ""}…
                  </span>
                )}
                {attachErr && (
                  <span className="flex items-center gap-1.5 text-[12px] px-2.5 py-1 rounded-lg" style={{ background: "rgba(191,97,106,.18)", color: "#e08", }}>
                    <X size={12} /> {attachErr}
                  </span>
                )}
              </div>
            )}
            <div className="flex items-end gap-2 rounded-xl px-3 py-2.5" style={{ background: "var(--surface2)", boxShadow: "var(--sh3)" }}>
              <input ref={fileInput} type="file" multiple className="hidden" onChange={e => { onFiles(e.target.files); e.target.value = "" }} />
              <button onClick={pickFiles} className={iconBtn} style={{ color: "var(--muted)" }}
                onMouseEnter={e => (e.currentTarget.style.background = "var(--surface)")} onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                title="Attach file"><Paperclip size={18} /></button>
              <textarea ref={ta} rows={1} value={input} placeholder={transcribing ? "Transcribing…" : "Message agent…"}
                onChange={e => { setInput(e.target.value); grow(e.target) }} onKeyDown={onKey}
                className="flex-1 resize-none bg-transparent outline-none text-[14.5px] leading-relaxed py-1.5 max-h-[200px]" style={{ color: "var(--ink)" }} />
              <button onClick={recording ? stopRec : startRec} className={iconBtn}
                style={{ color: recording ? "#fff" : "var(--muted)", background: recording ? "var(--accent)" : "transparent" }}
                onMouseEnter={e => { if (!recording) e.currentTarget.style.background = "var(--surface)" }}
                onMouseLeave={e => { if (!recording) e.currentTarget.style.background = "transparent" }}
                title={recording ? "Stop recording" : "Record voice"}>{recording ? <Square size={15} fill="#fff" /> : <Mic size={18} />}</button>
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
  const [searxng, setSearxng] = useState("")
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState("")
  useEffect(() => {
    (async () => {
      try {
        const c = await (await fetch("/settings")).json()
        setCfg(c); setProvider(c.provider || "openrouter"); setBaseUrl(c.base_url || "")
        setModel(c.model || ""); setCodeModel(c.code_model || ""); setDeepModel(c.deep_model || "")
        setWorkMode(c.work_mode || "auto-accept"); setForceMode(c.force_mode || ""); setSearxng(c.searxng_url || "")
      } catch { /* */ }
    })()
  }, [])
  const save = async () => {
    setSaving(true); setMsg("")
    try {
      const body: any = { provider, base_url: baseUrl, model, code_model: codeModel, deep_model: deepModel, work_mode: workMode, force_mode: forceMode, searxng_url: searxng }
      if (apiKey) body.api_key = apiKey
      const d = await (await fetch("/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json()
      setMsg(d.message || ""); setCfg(c => ({ ...c, active: d.active, api_key_source: d.api_key_source })); setApiKey("")
    } catch (e: any) { setMsg("error: " + e.message) } finally { setSaving(false) }
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
          <h2 className="text-[18px] font-bold tracking-tight" style={{ color: "var(--ink)" }}>Settings</h2>
          <button onClick={onClose} className="ml-auto grid place-items-center w-8 h-8 rounded-lg transition-colors" style={{ color: "var(--muted)" }}
            onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")} onMouseLeave={e => (e.currentTarget.style.background = "transparent")}><X size={18} /></button>
        </div>
        <div className="px-6 pb-4 overflow-y-auto space-y-4">
          <div><Lbl>Provider</Lbl>
            <Dropdown value={provider} onChange={setProvider} options={[{ v: "openrouter", l: "OpenRouter" }, { v: "ollama", l: "Ollama (local)" }]} /></div>
          <div className="grid grid-cols-2 gap-3">
            <div><Lbl>Main model</Lbl><input value={model} onChange={e => setModel(e.target.value)} placeholder="google/gemini-2.5-flash-lite" className={field} style={fst} /></div>
            <div><Lbl>Code model</Lbl><input value={codeModel} onChange={e => setCodeModel(e.target.value)} placeholder="deepseek/deepseek-v4-flash" className={field} style={fst} /></div>
          </div>
          <div><Lbl>Heavy review model (deep)</Lbl><input value={deepModel} onChange={e => setDeepModel(e.target.value)} placeholder="model for heavy mode" className={field} style={fst} /></div>
          <div><Lbl>Endpoint (base_url)</Lbl><input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="https://openrouter.ai/api/v1" className={field} style={fst} /></div>
          <div><Lbl>API key · now: {cfg.api_key_source || "—"}</Lbl>
            <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="paste key (optional)" className={field} style={fst} /></div>
          <div><Lbl>SearXNG · web &amp; image search</Lbl>
            <input value={searxng} onChange={e => setSearxng(e.target.value)} placeholder="http://localhost:8080 (your instance)" className={field} style={fst} />
            <div className="mt-1 text-[11px]" style={{ color: "var(--faint)" }}>Your SearXNG → stable search and images in chat. Empty → DuckDuckGo fallback.</div></div>
          <div className="grid grid-cols-2 gap-3">
            <div><Lbl>Work mode</Lbl>
              <Dropdown value={workMode} onChange={setWorkMode} options={[
                { v: "manual", l: "Ask" }, { v: "auto-accept", l: "Auto-accept" }, { v: "auto", l: "Autonomous" }]} /></div>
            <div><Lbl>Thinking mode</Lbl>
              <Dropdown value={forceMode} onChange={setForceMode} options={[
                { v: "", l: "Auto" }, { v: "fast", l: "Fast" }, { v: "reason", l: "Reason" },
                { v: "act", l: "Act" }, { v: "deliberate", l: "Deliberate" }, { v: "heavy", l: "Heavy" }]} /></div>
          </div>
          <div className="pt-1">
            <Lbl>Browser extension <span style={{ color: cfg.bridge_connected ? "#3ecf8e" : "var(--faint)", fontWeight: 600 }}>
              · {cfg.bridge_connected ? "connected" : "not connected"}</span></Lbl>
            <div className="rounded-lg p-3 text-[12px] leading-relaxed" style={{ background: "var(--surface)", color: "var(--muted)" }}>
              To let the agent act in YOUR Chrome (logins, tabs): <code className="px-1 rounded" style={{ background: "var(--hover)", fontFamily: "var(--font-mono)" }}>chrome://extensions</code> → Developer mode → Load unpacked → the <b>extension/</b> folder, paste the token into the popup:
              <div className="mt-2 px-2.5 py-1.5 rounded select-all break-all" style={{ background: "var(--hover)", color: "var(--ink)", fontFamily: "var(--font-mono)", fontSize: 11 }}>{cfg.bridge_token || "—"}</div>
            </div>
          </div>
          {cfg.active && <div className="text-[11.5px] leading-relaxed" style={{ color: "var(--faint)" }}>Active: {cfg.active}</div>}
          {msg && <div className="text-[12.5px] font-medium" style={{ color: "var(--accent)" }}>{msg}</div>}
        </div>
        <div className="px-6 py-4">
          <motion.button whileTap={{ scale: 0.98 }} onClick={save} disabled={saving}
            className="w-full py-2.5 rounded-lg text-[13.5px] font-semibold disabled:opacity-50"
            style={{ background: "var(--accent)", color: "var(--accent-fg)" }}>{saving ? "Checking…" : "Save & verify"}</motion.button>
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

function AnimatedDots() {
  return <span className="inline-flex ml-0.5">
    {[0, 1, 2].map(i => <motion.span key={i} animate={{ opacity: [0.15, 1, 0.15] }}
      transition={{ duration: 1.3, repeat: Infinity, delay: i * 0.22, ease: "easeInOut" }}>.</motion.span>)}
  </span>
}

// Реальный ход исполнения (прогресс по узлам графа) — траектория, текущий шаг ярче.
function Working({ steps, current }: { steps: string[]; current: string }) {
  const shown = (steps.length ? steps : current ? [current] : ["Thinking"]).slice(-6)
  return (
    <div className="py-1 space-y-1.5">
      <AnimatePresence initial={false}>
        {shown.map((s, i) => {
          const last = i === shown.length - 1
          return (
            <motion.div key={s} layout initial={{ opacity: 0, filter: "blur(5px)", x: -4 }}
              animate={{ opacity: last ? 1 : 0.4, filter: "blur(0px)", x: 0 }} transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
              className="flex items-center gap-2 text-[13.5px]" style={{ color: last ? "var(--ink)" : "var(--faint)" }}>
              {last
                ? <motion.span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: "var(--accent)" }}
                    animate={{ opacity: [1, 0.25, 1], scale: [1, 0.6, 1] }} transition={{ duration: 1.3, repeat: Infinity, ease: "easeInOut" }} />
                : <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: "var(--faint)" }} />}
              {last
                ? <span className="flex items-center">{s}<AnimatedDots /></span>
                : <span>{s}</span>}
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}

function ClarifyCard({ questions, onSubmit }: { questions: { question: string; options: string[]; why?: string }[]; onSubmit: (a: string[]) => void }) {
  const [sel, setSel] = useState<Record<number, string[]>>({})
  const [txt, setTxt] = useState<Record<number, string>>({})
  const toggle = (qi: number, opt: string) => setSel(s => {
    const cur = new Set(s[qi] || []); cur.has(opt) ? cur.delete(opt) : cur.add(opt); return { ...s, [qi]: [...cur] }
  })
  const submit = () => onSubmit(questions.map((_q, i) => [...(sel[i] || []), (txt[i] || "").trim()].filter(Boolean).join(", ")))
  const ok = questions.every((_q, i) => (sel[i]?.length || (txt[i] || "").trim()))
  return (
    <motion.div initial={{ opacity: 0, y: 10, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.4 }}
      className="rounded-xl p-5 mb-4" style={{ background: "var(--surface)", boxShadow: "var(--sh2)" }}>
      <div className="text-[11px] font-bold tracking-[0.1em] mb-4" style={{ color: "var(--accent)" }}>CLARIFY</div>
      <div className="space-y-5">
        {questions.map((q, i) => (
          <div key={i}>
            <div className="text-[14.5px] font-semibold mb-2.5" style={{ color: "var(--ink)" }}>{q.question}</div>
            {q.options?.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-2.5">
                {q.options.map(opt => {
                  const on = (sel[i] || []).includes(opt)
                  return (
                    <button key={opt} onClick={() => toggle(i, opt)}
                      className="px-3 py-1.5 rounded-lg text-[13px] font-medium border transition-colors flex items-center gap-1.5"
                      style={{ background: on ? "var(--accent)" : "transparent", color: on ? "var(--accent-fg)" : "var(--ink)", borderColor: on ? "var(--accent)" : "var(--bd2)" }}>
                      {on && <Check size={13} />}{opt}
                    </button>
                  )
                })}
              </div>
            )}
            <input value={txt[i] || ""} onChange={e => setTxt(s => ({ ...s, [i]: e.target.value }))}
              placeholder={q.options?.length ? "your option…" : "answer…"}
              className="w-full px-3 py-2 rounded-lg text-[13.5px] outline-none" style={{ background: "var(--sunken)", color: "var(--ink)" }} />
          </div>
        ))}
      </div>
      <motion.button whileTap={{ scale: 0.98 }} onClick={submit} disabled={!ok}
        className="mt-5 px-5 py-2.5 rounded-lg text-[13.5px] font-semibold disabled:opacity-40"
        style={{ background: "var(--accent)", color: "var(--accent-fg)" }}>Submit</motion.button>
    </motion.div>
  )
}

function ConfirmCard({ text, onConfirm }: { text: string; onConfirm: (v: string) => void }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.4 }}
      className="rounded-xl p-5 mb-4" style={{ background: "var(--surface)", boxShadow: "var(--sh2)" }}>
      <div className="text-[11px] font-bold tracking-[0.1em] mb-2.5" style={{ color: "var(--accent)" }}>CONFIRM</div>
      <div className="text-[14px] mb-4 leading-relaxed" style={{ color: "var(--ink)" }}>{text}</div>
      <div className="flex gap-2">
        <motion.button whileTap={{ scale: 0.97 }} onClick={() => onConfirm("yes")} className="px-5 py-2 rounded-lg text-[13.5px] font-semibold" style={{ background: "var(--accent)", color: "var(--accent-fg)" }}>Yes</motion.button>
        <motion.button whileTap={{ scale: 0.97 }} onClick={() => onConfirm("no")} className="px-5 py-2 rounded-lg text-[13.5px] font-semibold" style={{ background: "var(--sunken)", color: "var(--ink)" }}>No</motion.button>
      </div>
    </motion.div>
  )
}
