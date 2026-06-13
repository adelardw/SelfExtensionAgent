import { useEffect, useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, ArrowUp, Star, X, PanelLeftClose } from "lucide-react"

type Thread = { thread_id: string; title: string; favorite?: number }
type Msg = { role: "user" | "assistant"; content: string }

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
  const scroller = useRef<HTMLDivElement>(null)
  const ta = useRef<HTMLTextAreaElement>(null)

  const loadThreads = useCallback(async () => {
    try { const r = await fetch(`/chats?user_id=${uid}`); setThreads(await r.json()); setOnline(true) }
    catch { setOnline(false) }
  }, [])

  useEffect(() => { loadThreads(); const t = setInterval(loadThreads, 15000); return () => clearInterval(t) }, [loadThreads])
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight }) }, [msgs])

  const newChat = () => { setCur(newId()); setMsgs([]); setTitle("Новый чат"); setMode(""); ta.current?.focus() }

  const openThread = async (id: string) => {
    setCur(id)
    try {
      const d = await (await fetch(`/chats/${id}`)).json()
      setTitle(d.thread?.title || "Чат"); setMsgs(d.messages || [])
    } catch { setMsgs([]) }
  }

  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    await fetch(`/chats/${id}`, { method: "DELETE" })
    if (id === cur) newChat(); else loadThreads()
  }

  const send = async () => {
    const text = input.trim(); if (!text || busy) return
    setBusy(true); setInput("")
    setMsgs(m => [...m, { role: "user", content: text }, { role: "assistant", content: "…" }])
    try {
      const r = await fetch("/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: uid, thread_id: cur, query: text }),
      })
      if (!r.ok) throw new Error("HTTP " + r.status)
      const d = await r.json()
      setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: "assistant", content: d.answer || "(пусто)" }; return c })
      if (d.mode) setMode(d.mode)
      if (d.title) setTitle(d.title)
      loadThreads()
    } catch (e: any) {
      setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: "assistant", content: "⚠ ошибка: " + e.message }; return c })
    } finally { setBusy(false); ta.current?.focus() }
  }

  const onKey = (e: React.KeyboardEvent) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() } }
  const grow = (el: HTMLTextAreaElement) => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 200) + "px" }
  const prompts = ["Сравни 3 ноутбука до 100к и посоветуй", "Объясни attention в трансформере со ссылками", "Посчитай сложный процент по вкладу"]

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside className="flex flex-col shrink-0 border-r overflow-hidden transition-[width] duration-200"
        style={{ width: sideOpen ? 248 : 0, background: "var(--elev)", borderColor: "var(--bd)" }}>
        <div className="w-[248px] flex flex-col h-full">
          <div className="flex items-center gap-2.5 h-14 px-4 border-b shrink-0" style={{ borderColor: "var(--bd)" }}>
            <div className="grid place-items-center w-[22px] h-[22px] rounded-[5px] text-[11px] font-bold"
              style={{ background: "var(--inv)", color: "var(--inv-fg)" }}>S</div>
            <span className="text-[13px] font-semibold tracking-tight">self&#8209;extension</span>
            <span className="ml-auto text-[10px] font-mono" style={{ color: "var(--faint)" }}>0.2</span>
          </div>
          <div className="p-3">
            <button onClick={newChat}
              className="w-full flex items-center gap-2 px-3 py-2 text-[13px] font-medium rounded-md border transition-colors"
              style={{ borderColor: "var(--bd2)", color: "var(--fg)" }}
              onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")}
              onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
              <Plus size={15} /> Новый чат
              <kbd className="ml-auto text-[10px] font-mono" style={{ color: "var(--faint)" }}>⌘N</kbd>
            </button>
          </div>
          <div className="px-4 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--faint)" }}>История</div>
          <div className="flex-1 overflow-y-auto px-2 pb-3 min-h-0">
            {threads.length === 0 && <div className="px-2 py-2 text-[12.5px]" style={{ color: "var(--faint)" }}>пока пусто</div>}
            {threads.map(t => (
              <div key={t.thread_id} onClick={() => openThread(t.thread_id)}
                className="group relative flex items-center px-2.5 py-[7px] rounded-md cursor-pointer text-[13px] truncate transition-colors"
                style={{ background: t.thread_id === cur ? "var(--sel)" : "transparent", color: t.thread_id === cur ? "var(--fg)" : "var(--muted)" }}
                onMouseEnter={e => { if (t.thread_id !== cur) e.currentTarget.style.background = "var(--hover)" }}
                onMouseLeave={e => { if (t.thread_id !== cur) e.currentTarget.style.background = "transparent" }}>
                {!!t.favorite && <Star size={11} className="mr-1.5 shrink-0" fill="currentColor" />}
                <span className="truncate">{t.title || "без названия"}</span>
                <button onClick={e => del(t.thread_id, e)}
                  className="absolute right-1.5 opacity-0 group-hover:opacity-100 p-1 rounded transition-opacity"
                  style={{ color: "var(--faint)" }}><X size={12} /></button>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-2 px-4 py-3 border-t text-[10.5px] font-mono" style={{ borderColor: "var(--bd)", color: "var(--faint)" }}>
            <span className="w-[5px] h-[5px] rounded-full" style={{ background: online ? "#3ecf8e" : "var(--faint)" }} />
            {online ? uid : "offline"}
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex flex-col flex-1 min-w-0 min-h-0">
        <div className="flex items-center gap-3 h-14 px-7 border-b shrink-0" style={{ borderColor: "var(--bd)" }}>
          <button onClick={() => setSideOpen(s => !s)} className="grid place-items-center -ml-1 p-1.5 rounded-md transition-colors"
            style={{ color: "var(--muted)" }} onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}><PanelLeftClose size={17} /></button>
          <span className="text-[13px] font-medium truncate">{title}</span>
          {mode && <span className="ml-auto text-[11px] font-mono" style={{ color: "var(--faint)" }}>{mode}</span>}
        </div>

        <div ref={scroller} className="flex-1 overflow-y-auto min-h-0">
          {msgs.length === 0 ? (
            <div className="h-full flex flex-col justify-center max-w-3xl mx-auto px-7 w-full">
              <div className="grid place-items-center w-7 h-7 rounded-[5px] text-[13px] font-bold mb-6"
                style={{ background: "var(--inv)", color: "var(--inv-fg)" }}>S</div>
              <h1 className="text-[28px] font-semibold tracking-[-0.03em] leading-tight">С чего начнём?</h1>
              <p className="mt-3 text-[14px]" style={{ color: "var(--muted)" }}>Поиск, анализ, вычисления, браузер — и память о тебе.</p>
              <div className="mt-7 max-w-[460px] rounded-md border overflow-hidden" style={{ borderColor: "var(--bd)" }}>
                {prompts.map((p, i) => (
                  <button key={i} onClick={() => { setInput(p); setTimeout(() => ta.current && grow(ta.current), 0) }}
                    className="block w-full text-left px-4 py-3 text-[13px] border-b last:border-b-0 transition-colors"
                    style={{ borderColor: "var(--bd)", color: "var(--muted)" }}
                    onMouseEnter={e => { e.currentTarget.style.background = "var(--hover)"; e.currentTarget.style.color = "var(--fg)" }}
                    onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--muted)" }}>{p}</button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-7 py-10">
              <AnimatePresence initial={false}>
                {msgs.map((m, i) => (
                  <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="pb-8">
                    <div className="text-[10.5px] font-semibold uppercase tracking-[0.11em] mb-3"
                      style={{ color: m.role === "user" ? "var(--faint)" : "var(--muted)" }}>
                      {m.role === "user" ? "Ты" : "Агент"}
                    </div>
                    {m.role === "user"
                      ? <div className="text-[15px] leading-relaxed whitespace-pre-wrap">{m.content}</div>
                      : m.content === "…"
                        ? <Dots />
                        : <div className="md" dangerouslySetInnerHTML={{ __html: md(m.content) }} />}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>

        <div className="px-7 pb-6 pt-3">
          <div className="max-w-3xl mx-auto flex items-end gap-2.5 rounded-lg border px-4 py-2.5 transition-colors"
            style={{ background: "var(--card)", borderColor: "var(--bd2)" }}>
            <textarea ref={ta} rows={1} value={input} placeholder="Сообщение агенту…"
              onChange={e => { setInput(e.target.value); grow(e.target) }} onKeyDown={onKey}
              className="flex-1 resize-none bg-transparent outline-none text-[14px] leading-relaxed py-1 max-h-[200px]"
              style={{ color: "var(--fg)" }} />
            <button onClick={send} disabled={busy || !input.trim()}
              className="grid place-items-center w-7 h-7 rounded-md shrink-0 transition-opacity disabled:opacity-40"
              style={{ background: "var(--inv)", color: "var(--inv-fg)" }}><ArrowUp size={15} /></button>
          </div>
          <div className="max-w-3xl mx-auto mt-2 text-center text-[10.5px] font-mono" style={{ color: "var(--faint)" }}>
            Enter — отправить · Shift+Enter — перенос
          </div>
        </div>
      </main>
    </div>
  )
}

function Dots() {
  return <div className="flex gap-1 py-1">
    {[0, 1, 2].map(i => <motion.span key={i} className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--muted)" }}
      animate={{ opacity: [0.25, 1, 0.25] }} transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.18 }} />)}
  </div>
}
