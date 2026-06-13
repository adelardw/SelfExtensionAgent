import { useEffect, useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, ArrowUp, Star, X, PanelLeft } from "lucide-react"

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
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }) }, [msgs])

  const newChat = () => { setCur(newId()); setMsgs([]); setTitle("Новый чат"); setMode(""); ta.current?.focus() }
  const openThread = async (id: string) => {
    setCur(id)
    try { const d = await (await fetch(`/chats/${id}`)).json(); setTitle(d.thread?.title || "Чат"); setMsgs(d.messages || []) }
    catch { setMsgs([]) }
  }
  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); await fetch(`/chats/${id}`, { method: "DELETE" })
    if (id === cur) newChat(); else loadThreads()
  }
  const send = async () => {
    const text = input.trim(); if (!text || busy) return
    setBusy(true); setInput(""); if (ta.current) ta.current.style.height = "auto"
    setMsgs(m => [...m, { role: "user", content: text }, { role: "assistant", content: "…" }])
    try {
      const r = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ user_id: uid, thread_id: cur, query: text }) })
      if (!r.ok) throw new Error("HTTP " + r.status)
      const d = await r.json()
      setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: "assistant", content: d.answer || "(пусто)" }; return c })
      if (d.mode) setMode(d.mode); if (d.title) setTitle(d.title); loadThreads()
    } catch (e: any) {
      setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: "assistant", content: "⚠ ошибка: " + e.message }; return c })
    } finally { setBusy(false); ta.current?.focus() }
  }
  const onKey = (e: React.KeyboardEvent) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() } }
  const grow = (el: HTMLTextAreaElement) => { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 200) + "px" }
  const prompts = ["Сравни 3 ноутбука до 100к и посоветуй", "Объясни attention в трансформере со ссылками", "Посчитай сложный процент по вкладу"]

  return (
    <div className="flex h-full" style={{ background: "var(--bg)" }}>
      {/* Sidebar — поверхность светлее фона + мягкая тень: разделение цветом, не линией */}
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
          <div className="flex items-center gap-2 px-5 py-3.5 text-[11px] font-mono" style={{ color: "var(--faint)" }}>
            <motion.span className="w-[7px] h-[7px] rounded-full" style={{ background: online ? "#7e8a4e" : "var(--faint)" }}
              animate={online ? { opacity: [1, 0.5, 1] } : {}} transition={{ duration: 2.5, repeat: Infinity }} />
            {online ? uid : "offline"}
          </div>
        </div>
      </motion.aside>

      {/* Main */}
      <main className="flex flex-col flex-1 min-w-0 min-h-0">
        <div className="flex items-center gap-3 px-7 pt-5 pb-3">
          <button onClick={() => setSideOpen(s => !s)} className="grid place-items-center -ml-1.5 w-8 h-8 rounded-lg transition-colors"
            style={{ color: "var(--muted)" }} onMouseEnter={e => (e.currentTarget.style.background = "var(--surface)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}><PanelLeft size={18} /></button>
          <span className="text-[14px] font-semibold truncate" style={{ color: "var(--ink)" }}>{title}</span>
          {mode && <span className="ml-auto text-[11px] font-medium px-2.5 py-1 rounded-full" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>{mode}</span>}
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
                    <motion.button key={i} whileHover={{ y: -2 }} whileTap={{ scale: 0.99 }}
                      onClick={() => { setInput(p); setTimeout(() => ta.current && grow(ta.current), 0); ta.current?.focus() }}
                      className="text-left px-4 py-3.5 text-[13.5px] rounded-2xl transition-shadow"
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
                  <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: [0.22, 0.7, 0.2, 1] }} className="pb-7">
                    {m.role === "user" ? (
                      <div className="flex justify-end">
                        <div className="max-w-[85%] px-4 py-3 rounded-3xl rounded-br-lg text-[14.5px] leading-relaxed whitespace-pre-wrap"
                          style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: "var(--sh1)" }}>{m.content}</div>
                      </div>
                    ) : (
                      <div>
                        <div className="text-[11px] font-bold tracking-wide mb-2.5" style={{ color: "var(--faint)" }}>АГЕНТ</div>
                        {m.content === "…" ? <Dots /> : <div className="md" style={{ color: "var(--ink)" }} dangerouslySetInnerHTML={{ __html: md(m.content) }} />}
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>

        <div className="px-7 pb-7 pt-2">
          <div className="max-w-3xl mx-auto flex items-end gap-2.5 rounded-[20px] px-4 py-3 transition-shadow"
            style={{ background: "var(--surface2)", boxShadow: "var(--sh3)" }}>
            <textarea ref={ta} rows={1} value={input} placeholder="Сообщение агенту…"
              onChange={e => { setInput(e.target.value); grow(e.target) }} onKeyDown={onKey}
              className="flex-1 resize-none bg-transparent outline-none text-[14.5px] leading-relaxed py-1.5 max-h-[200px]" style={{ color: "var(--ink)" }} />
            <motion.button whileTap={{ scale: 0.9 }} onClick={send} disabled={busy || !input.trim()}
              className="grid place-items-center w-9 h-9 rounded-2xl shrink-0 transition-all disabled:opacity-35"
              style={{ background: "var(--accent)", color: "var(--accent-fg)", boxShadow: input.trim() ? "var(--sh1)" : "none" }}><ArrowUp size={17} strokeWidth={2.5} /></motion.button>
          </div>
        </div>
      </main>
    </div>
  )
}

function Dots() {
  return <div className="flex gap-1.5 py-1.5">
    {[0, 1, 2].map(i => <motion.span key={i} className="w-2 h-2 rounded-full" style={{ background: "var(--accent)" }}
      animate={{ opacity: [0.25, 1, 0.25], y: [0, -3, 0] }} transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.16 }} />)}
  </div>
}
