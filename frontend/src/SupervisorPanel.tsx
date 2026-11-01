import { useEffect, useState } from 'react'

type Agent = { id: string; role: string; status: string; task: string; reasoning: string; output: string; context: unknown[]; receipt: unknown; actions: {tool:string;ok:boolean;evidence:string}[] }
type Mind = { id:string; parent:string; mode:string; memory:{summary:string; validation:{stale_files:string[]; evidence_present:boolean}} }
type Proposal = {id:string;parent:string;status:string;files:string[];verification:{name:string;ok:boolean;evidence_id:string}[]}
type Obligations = {requirements?:{id:string;text:string;status:string}[];claims?:{id:string;text:string;status:string;evidence:string[];depends_on:string[];support_kind:string}[]}
type View = { proposals?:Proposal[]; profiles?:{name:string;base:string;custom:boolean}[]; cognition?:Mind[]; thinking?: boolean; context?: {obligations?:Obligations;navigation?:unknown}; supervisor?: { agents?: Agent[]; reviews?: number } }

export function SupervisorPanel({ runId, onClose }: { runId: string; onClose: () => void }) {
  const [view, setView] = useState<View>({})
  const [message, setMessage] = useState('')
  const [target, setTarget] = useState('main')
  const [error, setError] = useState('')
  const [mode, setMode] = useState('auto')
  const [context, setContext] = useState(false)
  useEffect(() => {
    let disposed = false
    const load = async () => {
      try {
        const response = await fetch(`/run/${runId}/context`)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        if (!disposed) setView(data)
      } catch (e) { if (!disposed) setError(String(e)) }
    }
    void load()
    const timer = setInterval(() => { void load() }, 1000)
    return () => { disposed = true; clearInterval(timer) }
  }, [runId])
  const steer = async (pause = false) => {
    try {
      const response = await fetch(`/run/${runId}/steer`, { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ message:pause ? 'Pause at the next safe boundary.' : message, target, action:pause ? 'pause' : 'steer', interrupt:true }) })
      if (!response.ok) throw new Error(await response.text())
      setMessage(''); setError('')
    } catch (e) { setError(String(e)) }
  }
  const switchMode = async () => {
    try {
      const response = await fetch(`/run/${runId}/steer`, {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action:'switch',target,mode})})
      if (!response.ok) throw new Error(await response.text())
      setError('')
    } catch(e) { setError(String(e)) }
  }
  const agents = view.supervisor?.agents || []
  return <div className="fixed inset-0 z-50 flex justify-end" style={{background:'#0006'}} onClick={onClose}>
    <section role="dialog" aria-modal="true" aria-label="Оператор агентов" onClick={e=>e.stopPropagation()}
      className="w-full max-w-xl h-full overflow-y-auto p-5 space-y-4" style={{background:'var(--surface)',color:'var(--ink)'}}>
      <div className="flex justify-between"><b>Оператор агентов</b><button onClick={onClose} aria-label="Закрыть">✕</button></div>
      <p className="text-xs" style={{color:'var(--muted)'}}>Наблюдение не вызывает LLM. Ветки имеют отдельный контекст. Правки готовятся в отдельных копиях проекта и проходят проверку перед принятием. Reasoning показывается, если его возвращает провайдер.</p>
      {!!view.context?.obligations?.requirements?.length && <details open className="text-xs space-y-2">
        <summary className="cursor-pointer">Требования и подтверждения</summary>
        {view.context.obligations.requirements.map(r=><p key={r.id}><b>{r.id} · {r.status}</b> — {r.text}</p>)}
        <p style={{color:'var(--muted)'}}>Supported — оценка по указанным данным. Успешный эксперимент не является общим доказательством.</p>
        {view.context.obligations.claims?.map(c=><div key={c.id} className="rounded p-2" style={{background:'var(--surface2)'}}>
          <b>{c.id} · {c.status}</b><p>{c.text}</p><p>{c.support_kind} · evidence {c.evidence.join(', ') || 'нет'}{c.depends_on.length ? ` · зависит от ${c.depends_on.join(', ')}` : ''}</p>
        </div>)}
      </details>}
      {view.cognition?.map(m=><div key={m.id} className="text-xs rounded p-2" style={{background:'var(--surface2)'}}>
        <b>{m.parent ? `${m.parent} → ` : ''}{m.id} · {m.mode}</b>
        <p>{m.memory.summary || 'Сводка пока не сформирована'}</p>
        <p>Evidence: {m.memory.validation.evidence_present?'на месте':'неполный'} · устаревших файлов: {m.memory.validation.stale_files.length}. Сводка требует проверки.</p>
      </div>)}
      {view.proposals?.filter(p=>p.files.length>0).map(p=><article key={p.id} className="rounded p-3 text-xs space-y-1" style={{background:'var(--surface2)'}}>
        <b>Патч {p.id} → {p.parent} · {p.status}</b><p>{p.files.join(', ')}</p>
        {p.verification?.map(c=><p key={c.name}>{c.ok?'✓':'×'} {c.name} · evidence {c.evidence_id}</p>)}
        {p.status==='proposed' && <button className="underline" onClick={()=>{setTarget(p.parent);setMessage(`Посмотри patch ${p.id}, проверь соответствие задаче и прими через accept, если всё верно.`)}}>Направить родителю на проверку</button>}
      </article>)}
      {agents.map(a=><article key={a.id} className="rounded-lg p-3 space-y-2" style={{background:'var(--surface2)'}}>
        <div className="flex justify-between text-sm"><b>{a.id} · {a.role}</b><span>{a.status}</span></div>
        <p className="text-xs">{a.task}</p>
        {a.reasoning && <details open={a.status==='thinking'}><summary className="text-xs cursor-pointer">Reasoning · live</summary><pre className="text-xs whitespace-pre-wrap max-h-48 overflow-y-auto">{a.reasoning}</pre></details>}
        <div className="text-xs">{a.actions.map((x,i)=><div key={i}>{x.ok?'✓':'×'} {x.tool} · evidence {x.evidence}</div>)}</div>
        <details><summary className="text-xs cursor-pointer">Фактический контекст запроса</summary><pre className="text-xs whitespace-pre-wrap max-h-64 overflow-y-auto">{JSON.stringify({receipt:a.receipt,messages:a.context},null,2)}</pre></details>
      </article>)}
      <button className="text-sm underline" onClick={()=>setContext(!context)}>Какие примеры и находки попали в контекст</button>
      {context && <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(view.context,null,2)}</pre>}
      <label className="block text-sm">Кому
        <select value={target} onChange={e=>setTarget(e.target.value)} className="block w-full p-2 rounded mt-1" style={{background:'var(--surface2)'}}>
          <option value="main">Основному агенту</option>{agents.filter(a=>a.id!=='main' && a.id!=='supervisor' && !a.id.startsWith('decision-') && !a.id.startsWith('critic-') && !a.id.startsWith('observer-')).map(a=><option key={a.id}>{a.id}</option>)}
        </select></label>
      {!!view.cognition?.length && <div className="flex gap-2 text-sm">
        <select aria-label="Тип мышления" value={mode} onChange={e=>setMode(e.target.value)} className="p-2 rounded" style={{background:'var(--surface2)'}}>
          {['auto',...(view.profiles?.map(p=>p.name) || ['inspect','reason','plan','research','review','repair','implement','verify'])].map(m=><option key={m}>{m}</option>)}
        </select><button onClick={()=>{void switchMode()}}>Сменить, сохранив контекст</button>
      </div>}
      <textarea value={message} onChange={e=>setMessage(e.target.value)} rows={3} placeholder="Уточнить задачу, исправить направление…" className="w-full p-2 rounded text-sm" style={{background:'var(--surface2)'}} />
      <div className="flex gap-3"><button disabled={!message.trim()} onClick={()=>{void steer()}} className="px-3 py-2 rounded text-sm" style={{background:'var(--accent)',color:'var(--accent-fg)'}}>Направить</button>
        <button onClick={()=>{void steer(true)}} className="text-sm underline">Приостановить</button></div>
      {error && <p className="text-xs" role="alert">{error}</p>}
    </section>
  </div>
}
