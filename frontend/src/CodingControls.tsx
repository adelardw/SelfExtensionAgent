import { useState } from "react"

import type { CodingOptions } from "./coding-options"

const efforts = [
  ["auto", "Авто", "Быстрое исполнение; больше рассуждений при планировании и повторных ошибках."],
  ["low", "Быстро", "Минимум рассуждений; фактический thinking зависит от модели."],
  ["medium", "Вдумчиво", "Thinking для анализа и разработки, ограниченный вывод."],
  ["high", "Глубоко", "Больше места для рассуждения и сложных исправлений."],
] as const

export function CodingControls({ value, onChange, busy, onResume }: {
  value: CodingOptions; onChange: (value: CodingOptions) => void; busy: boolean; onResume: () => void
}) {
  const [open, setOpen] = useState(false)
  const update = (patch: Partial<CodingOptions>) => {
    const next = { ...value, ...patch }
    onChange(next)
    try { localStorage.setItem("sea_coding_options", JSON.stringify(next)) } catch { /* optional persistence */ }
  }
  return <>
    <button disabled={busy} onClick={() => setOpen(true)} className="text-[12px] px-3 py-1.5 rounded-lg disabled:opacity-40"
      style={{ background: "var(--surface2)", color: "var(--ink)" }} aria-haspopup="dialog">
      {value.task_mode === "coding" ? "Движок" : "Ассистент"} · {efforts.find(e => e[0] === value.reasoning)?.[1]} ▾
    </button>
    {open && <div className="fixed inset-0 z-50 grid place-items-center p-4" style={{ background: "#0008" }} onClick={() => setOpen(false)}>
      <section role="dialog" aria-modal="true" aria-labelledby="coding-title" onClick={e => e.stopPropagation()}
        onKeyDown={e => { if (e.key === "Escape") setOpen(false) }}
        className="w-full max-w-md rounded-2xl p-6 space-y-4" style={{ background: "var(--surface)", color: "var(--ink)" }}>
        <div className="flex justify-between items-center"><h2 id="coding-title" className="font-semibold">Режим и глубина рассуждения</h2>
          <button onClick={() => setOpen(false)} aria-label="Закрыть">✕</button></div>
        <label className="block text-sm">Контур
          <select className="block w-full rounded p-2 mt-1" style={{ background: "var(--surface2)" }} value={value.task_mode}
            onChange={e => update({ task_mode: e.target.value as CodingOptions["task_mode"] })}>
            <option value="coding">Движок · код, математика, алгоритмы</option><option value="general">Ассистент · прежний граф</option>
          </select></label>
        {value.task_mode === "coding" ? <>
          <fieldset className="space-y-2"><legend className="text-sm mb-2">Рассуждение</legend>{efforts.map(([id, label, hint]) =>
            <label key={id} className="flex gap-3 cursor-pointer p-2 rounded" style={{ background: value.reasoning === id ? "var(--surface2)" : "transparent" }}>
              <input type="radio" name="reasoning" checked={value.reasoning === id} onChange={() => update({ reasoning: id })} />
              <span><b className="text-sm">{label}</b><span className="block text-xs" style={{ color: "var(--muted)" }}>{hint}</span></span>
            </label>)}</fieldset>
          <label className="block text-sm">Папка проекта
            <input value={value.workspace} onChange={e => update({ workspace: e.target.value })} placeholder="Текущая папка сервера"
              className="block w-full rounded p-2 mt-1 text-sm" style={{ background: "var(--surface2)" }} /></label>
          <label className="block text-sm">Исполнение
            <select value={value.execution} onChange={e => update({ execution: e.target.value as CodingOptions["execution"] })}
              className="block w-full rounded p-2 mt-1" style={{ background: "var(--surface2)" }}>
              <option value="auto">План → правки → тесты → исправления</option><option value="plan">Только план · без изменения файлов и команд</option>
            </select></label>
          <p className="text-xs" style={{ color: "var(--muted)" }}>Сейчас выбран дешёвый OpenRouter; локальный профиль ≤10B можно включить в конфигурации. Подтверждения определяются настройками доступа. Сабагенты включаются по необходимости в пределах общего бюджета.</p>
          <button disabled={busy} onClick={() => { setOpen(false); onResume() }} className="text-sm underline">Продолжить сохранённую задачу</button>
        </> : <p className="text-sm" style={{ color: "var(--muted)" }}>Провайдер и режимы универсального графа задаются в настройках.</p>}
        <button onClick={() => setOpen(false)} className="block w-full rounded-lg p-2 text-sm font-medium"
          style={{ background: "var(--accent)", color: "var(--accent-fg)" }}>Готово</button>
      </section>
    </div>}
  </>
}
