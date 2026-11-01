export type CodingOptions = {
  task_mode: "general" | "coding"
  reasoning: "auto" | "low" | "medium" | "high"
  execution: "auto" | "plan"
  workspace: string
}

export function initialCodingOptions(): CodingOptions {
  const defaults: CodingOptions = { task_mode: "coding", reasoning: "auto", execution: "auto", workspace: "" }
  try {
    const saved = JSON.parse(localStorage.getItem("sea_coding_options") || "{}")
    return { ...defaults,
      task_mode: saved.task_mode === "general" ? "general" : "coding",
      reasoning: ["auto", "low", "medium", "high"].includes(saved.reasoning) ? saved.reasoning : "auto",
      execution: saved.execution === "plan" ? "plan" : "auto",
      workspace: typeof saved.workspace === "string" ? saved.workspace : "" }
  } catch { return defaults }
}
