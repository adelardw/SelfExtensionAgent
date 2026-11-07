# Карта модулей SEA

Статический снимок Python-модулей на 5 ноября 2026. Docstring — описание автора; наличие функции не доказывает её корректность. Строки включают комментарии. Проверки и ограничения описаны в [разборе архитектуры](harness_review_20261105.md).

| Модуль | Строк | Символов | Назначение из исходника |
|---|---:|---:|---|
| [src/__init__.py](../src/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/agent_train/__init__.py](../src/agent_train/__init__.py) | 1 | 0 | Optional local training tools; importing this package never loads a model. |
| [src/agent_train/__main__.py](../src/agent_train/__main__.py) | 4 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/agent_train/bootstrap.py](../src/agent_train/bootstrap.py) | 44 | 1 | Verified SCRIPTED demonstrations for SFT; never on-policy RL rollouts. |
| [src/agent_train/cli.py](../src/agent_train/cli.py) | 77 | 3 | Training entry point; defaults to a config-only dry run. |
| [src/agent_train/config.py](../src/agent_train/config.py) | 156 | 8 | Strict YAML configs without importing MLX or starting inference. |
| [src/agent_train/dataset.py](../src/agent_train/dataset.py) | 83 | 2 | Audit traces and export only explicitly training-eligible, externally checked runs. |
| [src/agent_train/episode_grpo.py](../src/agent_train/episode_grpo.py) | 201 | 11 | On-policy, multi-action GRPO with outcome-only rewards and action-only loss. |
| [src/agent_train/episodes.py](../src/agent_train/episodes.py) | 225 | 17 | Small file-based tool curriculum, with a verifier outside the agent's tools. |
| [src/agent_train/grpo.py](../src/agent_train/grpo.py) | 166 | 7 | Bounded on-policy GRPO/QLoRA pilot for verified arithmetic code edits on MLX. |
| [src/agent_train/preflight.py](../src/agent_train/preflight.py) | 24 | 1 | Filesystem-only checks before loading the optional GPU backend. |
| [src/agent_train/sft.py](../src/agent_train/sft.py) | 95 | 4 | Audited chat SFT through MLX-LM; imported GPU dependencies only on execution. |
| [src/agent_train/trajectory.py](../src/agent_train/trajectory.py) | 163 | 14 | Local, bounded training telemetry. An episode outcome is never a reward by itself. |
| [src/agent_train/verified_edits.py](../src/agent_train/verified_edits.py) | 95 | 6 | A bounded single-edit RLVR environment. It is not a SWE benchmark. |
| [src/browser/__init__.py](../src/browser/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/browser/browser_bridge.py](../src/browser/browser_bridge.py) | 432 | 34 | Мост агент ↔ браузерное расширение (контур «агент живёт в ТВОЁМ браузере»). |
| [src/browser/browser_session.py](../src/browser/browser_session.py) | 364 | 27 | Живой управляемый браузер агента (контур «агент может делать в браузере ВСЁ»). |
| [src/coding/__init__.py](../src/coding/__init__.py) | 1 | 0 | Compact, local-first software development harness. |
| [src/coding/answer_verification.py](../src/coding/answer_verification.py) | 272 | 15 | Bind independently computed oracle values to the exact delivered JSON answer. |
| [src/coding/artifacts.py](../src/coding/artifacts.py) | 77 | 1 | Cheap format checks on delivered files; generation in a scratch copy is not delivery. |
| [src/coding/assertion_repair.py](../src/coding/assertion_repair.py) | 146 | 10 | Experimental mechanical binding of blind predictions to NEW scalar assertions. |
| [src/coding/attention.py](../src/coding/attention.py) | 84 | 7 | Durable human questions, inspired by DSH superagent's Attention lifecycle. |
| [src/coding/branches.py](../src/coding/branches.py) | 333 | 17 | Isolated branch snapshots and checked, conflict-aware patch adoption. |
| [src/coding/budget.py](../src/coding/budget.py) | 113 | 15 | Shared request admission across actors, critics, routers and retrieval. |
| [src/coding/cli.py](../src/coding/cli.py) | 62 | 3 | `sea solve` / `sea code`: one evidence-driven loop for software, mathematics and algorithms. |
| [src/coding/completion.py](../src/coding/completion.py) | 50 | 5 | Prepare a delivery candidate after checks; normal independent review still gates it. |
| [src/coding/containment.py](../src/coding/containment.py) | 123 | 5 | Host command confinement. Source edits use scoped edit/write tools; commands get scratch space. |
| [src/coding/context.py](../src/coding/context.py) | 185 | 20 | Project-local evidence board and bounded, verified experience retrieval. |
| [src/coding/continuation.py](../src/coding/continuation.py) | 205 | 12 | Short-context execution with an actor-local, selectively readable state archive. |
| [src/coding/contract_sources.py](../src/coding/contract_sources.py) | 61 | 6 | Bind requirement citations to exact input spans without model transcription. |
| [src/coding/control.py](../src/coding/control.py) | 116 | 13 | Human steering with cancellation only at model-generation boundaries. |
| [src/coding/critic.py](../src/coding/critic.py) | 847 | 30 | Bounded semantic review of task alignment and evidence; never an oracle of correctness. |
| [src/coding/decision.py](../src/coding/decision.py) | 167 | 9 | Typed cognitive routing. Remote router receives compact state, never tool transcripts. |
| [src/coding/decisions_api.py](../src/coding/decisions_api.py) | 131 | 7 | Bounded OpenRouter Decisions transport shared by retrieval and orchestration. |
| [src/coding/delivery.py](../src/coding/delivery.py) | 125 | 8 | Parse an explicit JSON key-list notation; leave arbitrary prose to the contract model. |
| [src/coding/dialogue.py](../src/coding/dialogue.py) | 201 | 13 | Bounded native dialogue, local to one actor and one steering version. |
| [src/coding/edits.py](../src/coding/edits.py) | 37 | 1 | Bounded coherent edits through the existing per-file execution boundary. |
| [src/coding/failure_diagnosis.py](../src/coding/failure_diagnosis.py) | 191 | 7 | One bounded recovery hypothesis; never completion evidence or tool authority. |
| [src/coding/followthrough.py](../src/coding/followthrough.py) | 68 | 2 | Turn acceptance gaps into a small, revision-aware work queue, without inference. |
| [src/coding/harness.py](../src/coding/harness.py) | 1299 | 28 | One model decision per step; deterministic verification and bounded context. |
| [src/coding/locking.py](../src/coding/locking.py) | 40 | 1 | One writer per workspace, including across CLI and desktop processes. |
| [src/coding/minds.py](../src/coding/minds.py) | 242 | 16 | Cognitive profiles, isolated working memory and bounded logical concurrency. |
| [src/coding/model.py](../src/coding/model.py) | 649 | 32 | Explicit local/cheap-cloud structured turns, without implicit provider fallback. |
| [src/coding/native_tools.py](../src/coding/native_tools.py) | 164 | 6 | Compile the same bounded Turn protocol to native function calls, without executing them. |
| [src/coding/navigation.py](../src/coding/navigation.py) | 131 | 8 | Bounded hierarchical retrieval with an unpruned frontier and explicit backtracking. |
| [src/coding/numerics.py](../src/coding/numerics.py) | 108 | 4 | Exact symbolic calculations with an AST interpreter; expressions are never eval'd. |
| [src/coding/obligations.py](../src/coding/obligations.py) | 161 | 12 | Persistent requirement/claim/evidence DAG. Support is scoped evidence, never truth. |
| [src/coding/orchestration.py](../src/coding/orchestration.py) | 735 | 29 | Task-created profiles, recursive workers, isolated edits and bounded live supervision. |
| [src/coding/output_recovery.py](../src/coding/output_recovery.py) | 48 | 6 | Bounded reuse of provider-returned draft text, never invented hidden reasoning. |
| [src/coding/probe_repair.py](../src/coding/probe_repair.py) | 59 | 6 | Bounded repair of a verifier program, without silently discarding its failure. |
| [src/coding/profiles.py](../src/coding/profiles.py) | 89 | 11 | Task-created thinking profiles; runtime capabilities remain independently bounded. |
| [src/coding/progress.py](../src/coding/progress.py) | 127 | 13 | Bounded exploration: repeated unchanged proposals temporarily leave the action menu. |
| [src/coding/project.py](../src/coding/project.py) | 70 | 6 | Small typed project notes and SWE guidance, shaped by explicit interaction evidence. |
| [src/coding/prompts.py](../src/coding/prompts.py) | 145 | 3 | Compact role prompts: disclose only usable tools; retain a fixed instruction hierarchy. |
| [src/coding/provider_errors.py](../src/coding/provider_errors.py) | 48 | 3 | Bounded diagnostics for rejected requests; never replay partial generations. |
| [src/coding/reference_repair.py](../src/coding/reference_repair.py) | 154 | 6 | Bounded, opt-in repair of agent-authored scalar test expectations. |
| [src/coding/request.py](../src/coding/request.py) | 218 | 11 | Structural context compaction and receipts for what actually reaches the model. |
| [src/coding/result_memory.py](../src/coding/result_memory.py) | 75 | 5 | Bounded, evidence-linked observations that survive the rolling action window. |
| [src/coding/source_prefetch.py](../src/coding/source_prefetch.py) | 61 | 1 | Bounded initial source observations, with no model calls or extra authority. |
| [src/coding/supervisor.py](../src/coding/supervisor.py) | 132 | 11 | Event-driven operator view. Observation is free; model review runs only on anomalies. |
| [src/coding/terminal.py](../src/coding/terminal.py) | 159 | 9 | Interactive developer console: commands remain available while the agent works. |
| [src/coding/test_provenance.py](../src/coding/test_provenance.py) | 111 | 12 | Session-local source history for narrowly scoped repair of agent-added tests. |
| [src/coding/test_trace.py](../src/coding/test_trace.py) | 134 | 2 | Observe the actual failing unittest, without asking a model to recreate its steps. |
| [src/coding/tool_scope.py](../src/coding/tool_scope.py) | 56 | 2 | Progressive tool disclosure: state-dependent menus within existing authority. |
| [src/coding/trajectory.py](../src/coding/trajectory.py) | 2 | 0 | Compatibility import; implementation moved to src.agent_train.trajectory. |
| [src/coding/trajectory_dataset.py](../src/coding/trajectory_dataset.py) | 2 | 0 | Compatibility import; use src.agent_train.dataset. |
| [src/coding/verifier.py](../src/coding/verifier.py) | 96 | 4 | Active review tools. Experiments run in a disposable snapshot, never the artifact. |
| [src/coding/worktools.py](../src/coding/worktools.py) | 241 | 1 | Bounded developer tools and explicit evidence sharing, independent of the loop. |
| [src/config/__init__.py](../src/config/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/config/cli_config.py](../src/config/cli_config.py) | 85 | 4 | CLI-настройки: `config.local.yml` МЕРДЖИТСЯ поверх `config.yml` (запрос юзера: «CLI должна иметь конфиг отдельный либо метчиться с config.yaml; изменения в CLI вписываются в конфиг»). |
| [src/config/config_paths.py](../src/config/config_paths.py) | 107 | 7 | Резолвер путей конфига — чтобы `sea`, установленный ПАКЕТОМ (глобально через `uv tool`/`pipx` или в venv через `uv pip install`), работал в ЛЮБОМ каталоге, а не только в исходном репозитории. |
| [src/config/core_config.py](../src/config/core_config.py) | 32 | 2 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/data/__init__.py](../src/data/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/data/knowledge_base.py](../src/data/knowledge_base.py) | 339 | 29 | База знаний пользователя: документы, которые юзер прикладывает (CLI/приложение), в ИЕРАРХИИ папок/подпапок, с retrieval. Агент отвечает из СОБСТВЕННЫХ документов юзера (персонализация + контекстный инжиниринг: в контекст — релевантные куски |
| [src/data/lightrag_engine.py](../src/data/lightrag_engine.py) | 121 | 8 | Движок графовой БЗ на НАСТОЯЩЕМ LightRAG (пакет lightrag-hku), не самодельный граф. Per-user инстанс: рабочая директория, LLM=наш OpenRouter (fast), эмбеддинги=OpenRouter. Ингест строит граф сущностей+связей; запрос — гибридный (граф + вект |
| [src/data/mcp_client.py](../src/data/mcp_client.py) | 264 | 10 | Автоподключение к MCP-серверам и использование их инструментов. |
| [src/data/research.py](../src/data/research.py) | 273 | 11 | Agentic research-слой: дисциплинированный МНОГОШАГОВЫЙ поиск с ВЕРИФИКАЦИЕЙ промежуточных фактов — а не наивный ReAct, который часто упирался в бюджет/пасовал на multi-hop. |
| [src/eval/__init__.py](../src/eval/__init__.py) | 3 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/eval/assistantbench_runner.py](../src/eval/assistantbench_runner.py) | 117 | 5 | AssistantBench — held-out бенч РЕАЛЬНЫХ повседневных веб-задач (студии рядом, хайки, вероятность дождя…). Сложнее GAIA для локального поиска: потолок <25% даже у топ-веб-агентов. |
| [src/eval/daily_eval.py](../src/eval/daily_eval.py) | 315 | 3 | Повседневный eval: прогоняет РЕАЛЬНЫЙ граф на типичных сценариях Mac-пользователя и собирает статистику бюджет/качество. |
| [src/eval/gaia_runner.py](../src/eval/gaia_runner.py) | 226 | 8 | GAIA — held-out бенчмарк (validation Level 1, текстовые задачи). |
| [src/eval/harness.py](../src/eval/harness.py) | 148 | 7 | Eval-харнесс: метрики качества агента + честная проверка, что backward улучшает. |
| [src/eval/harness_search.py](../src/eval/harness_search.py) | 364 | 21 | Provenance checks for offline harness comparison and legacy-log inventory. |
| [src/eval/judges.py](../src/eval/judges.py) | 155 | 2 | БИБЛИОТЕКА СУДЕЙ: персоны-валидаторы для мульти-агентных раундов — в репо, а не в чате. |
| [src/eval/pareto.py](../src/eval/pareto.py) | 193 | 9 | Offline quality/token selection inspired by ParetoSaddler (arXiv:2611.02079). |
| [src/eval/quality_cases.py](../src/eval/quality_cases.py) | 242 | 5 | Parameterized development tasks and host-only acceptance oracles. |
| [src/eval/quality_transfer_cases.py](../src/eval/quality_transfer_cases.py) | 131 | 3 | Additional families, written before evaluating the frozen answer-binding harness. |
| [src/eval/remote_files.py](../src/eval/remote_files.py) | 65 | 2 | Small file-tool program executed inside Harbor containers, not on the host. |
| [src/eval/route_eval.py](../src/eval/route_eval.py) | 614 | 2 | Статистическая оценка universal intent-роутера (src/intent.py) на РАЗМЕЧЕННОМ наборе. |
| [src/eval/sim_runner.py](../src/eval/sim_runner.py) | 207 | 14 | Ядро тестового стенда: ОДИН ход диалога с живым графом + ТЕЛЕМЕТРИЯ хода. |
| [src/eval/swebench_runner.py](../src/eval/swebench_runner.py) | 112 | 5 | SWE-bench Lite харнесс (настоящий, признанный бенчмарк). |
| [src/eval/terminalbench_agent.py](../src/eval/terminalbench_agent.py) | 160 | 16 | Harbor external-agent adapter: same SEA loop, shell tools in the task container. |
| [src/eval/user_sim.py](../src/eval/user_sim.py) | 188 | 6 | Мульти-агентная симуляция пользователей: LLM-персоны с уникальным профилем/стилем ведут МНОГОХОДОВЫЕ диалоги с реальным графом агента, реагируя на его настоящие ответы (уникальный UX-фидбек на каждом ходе), затем судья оценивает весь диалог |
| [src/external/__init__.py](../src/external/__init__.py) | 15 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/external/context.py](../src/external/context.py) | 53 | 6 | Контекст внешнего агента (A2A / MCP) в состоянии графа. |
| [src/graph/__init__.py](../src/graph/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/graph/agent.py](../src/graph/agent.py) | 3272 | 94 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/graph/intent.py](../src/graph/intent.py) | 438 | 19 | Универсальный (любой язык) embedding-kNN роутер интентов — «кодбук правильных роутов». |
| [src/graph/schemas.py](../src/graph/schemas.py) | 88 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/graph/semantic_signals.py](../src/graph/semantic_signals.py) | 403 | 26 | Анти-галлюцинация БЕЗ регэкспов/ключевых слов. |
| [src/graph/subagents.py](../src/graph/subagents.py) | 85 | 6 | Под-агенты и под-графы как ИНСТРУМЕНТЫ (agent/graph-as-tool). |
| [src/harness/__init__.py](../src/harness/__init__.py) | 7 | 0 | Public entry point for the shared mathematics, algorithms and software engine. |
| [src/improve/__init__.py](../src/improve/__init__.py) | 26 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/improve/__main__.py](../src/improve/__main__.py) | 58 | 1 | CLI для self-learning пайпа. |
| [src/improve/graph_learn.py](../src/improve/graph_learn.py) | 264 | 11 | Graph-aware batch self-improvement — «обучение» графа агента. |
| [src/improve/optimizer.py](../src/improve/optimizer.py) | 140 | 14 | Оптимизаторы промптов для self-learning пайпа. |
| [src/improve/pipe.py](../src/improve/pipe.py) | 259 | 17 | SelfLearningPipe — evolutionary-контур (L2 self-improvement). |
| [src/improve/prompt_store.py](../src/improve/prompt_store.py) | 262 | 19 | ParamStore — реестр обучаемых параметров графа (мульти-артефактный). |
| [src/improve/safety.py](../src/improve/safety.py) | 264 | 16 | Защита само-обучения от отравления (training-poisoning). |
| [src/interface/__init__.py](../src/interface/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/interface/chat_store.py](../src/interface/chat_store.py) | 200 | 12 | chat_store — постоянный индекс ЧАТОВ (тредов) поверх чекпоинтера графа. |
| [src/interface/clarify.py](../src/interface/clarify.py) | 181 | 14 | Реестр уточнений (clarification ledger) — онбординг неясных запросов как СВОЙСТВО СИСТЕМЫ, а не отдельная нода. |
| [src/interface/cli.py](../src/interface/cli.py) | 177 | 2 | Лёгкая точка входа консольной команды `sea` (см. pyproject [project.scripts]). |
| [src/interface/cli_art.py](../src/interface/cli_art.py) | 149 | 9 | ASCII-арт и анимации CLI: лого SEA (перелив синий↔металлик-серый), Squid-Girl (Ika Musume — морская тема SEA) и океан-спиннер (бегущая волна во время размышления). |
| [src/interface/compact.py](../src/interface/compact.py) | 166 | 14 | Сжатие контекста CLI (`/compact`) + статус-бар занятости контекстного окна + `COMPACT.md`. |
| [src/interface/interaction.py](../src/interface/interaction.py) | 83 | 6 | Журнал взаимодействий (interaction ledger) — стадия «↓ сигнал» замкнутого контура. |
| [src/interface/progress.py](../src/interface/progress.py) | 87 | 4 | Живой прогресс прогона графа: человекочитаемые статусы по нодам. |
| [src/interface/semantics.py](../src/interface/semantics.py) | 109 | 4 | Семантика ответов человека на подтверждения/вопросы агента. |
| [src/interface/server.py](../src/interface/server.py) | 910 | 51 | FastAPI-сервер: единый вход для всех клиентов (ПК, телефон, Telegram). |
| [src/interface/tui.py](../src/interface/tui.py) | 1178 | 76 | Full-screen TUI (Textual) — ДЕФОЛТНЫЙ интерфейс `sea`. |
| [src/llm/__init__.py](../src/llm/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/llm/llm.py](../src/llm/llm.py) | 286 | 20 | Единая точка для LLM-клиентов. Провайдер в config.yml: `provider` (openrouter  /  ollama). Модели выбираются по РОЛИ (fast  /  code  /  embed), а не по имени — без догадок и костылей. Фолбэк: provider=ollama → используем Ollama, если он реа |
| [src/llm/prompts.py](../src/llm/prompts.py) | 960 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/llm/structured_outputs.py](../src/llm/structured_outputs.py) | 325 | 24 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/llm/usage.py](../src/llm/usage.py) | 108 | 12 | Учёт расхода токенов LLM через LangChain-callback + персистентная статистика. |
| [src/maintenance/__init__.py](../src/maintenance/__init__.py) | 3 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/maintenance/__main__.py](../src/maintenance/__main__.py) | 18 | 1 | python -m src.maintenance            # обновить зависимости (с откатом при поломке) python -m src.maintenance --dry-run  # показать, что обновилось бы |
| [src/maintenance/db_prune.py](../src/maintenance/db_prune.py) | 44 | 1 | Прунинг checkpoints.db (найдено `sea doctor`: 258МБ на 531 тред). |
| [src/maintenance/dep_update.py](../src/maintenance/dep_update.py) | 87 | 3 | Авто-обновление зависимостей с проверкой и откатом. |
| [src/maintenance/doctor.py](../src/maintenance/doctor.py) | 442 | 20 | `sea doctor` — диагностика ВСЕХ известных болей одним прогоном. |
| [src/media.py](../src/media.py) | 442 | 26 | Работа с медиа-вложениями: картинки (vision) и голос (транскрипция). |
| [src/memory/__init__.py](../src/memory/__init__.py) | 16 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/memory/embedder.py](../src/memory/embedder.py) | 141 | 11 | Embedder — точка расширения для семантического поиска по памяти. |
| [src/memory/feedback.py](../src/memory/feedback.py) | 78 | 3 | Implicit Feedback — оценка удовлетворённости пользователя без явных оценок. |
| [src/memory/memory_tools.py](../src/memory/memory_tools.py) | 97 | 8 | Память-как-TOOL: агент САМ решает, когда подтянуть память (видение юзера «память превращается в инструмент»), а не только авто-впрыск memory_context в recall_node. |
| [src/memory/project_memory.py](../src/memory/project_memory.py) | 151 | 10 | Проектный ярус памяти (MEMORY.md) — порт паттерна memdir из backup (#2). |
| [src/memory/store.py](../src/memory/store.py) | 928 | 48 | MemoryStore — постоянная память агента поверх SQLite. |
| [src/memory/vector_index.py](../src/memory/vector_index.py) | 89 | 8 | VectorIndex — обёртка над TurboVec (IdMapIndex) для семантического recall. |
| [src/runtime/__init__.py](../src/runtime/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/runtime/artifacts.py](../src/runtime/artifacts.py) | 162 | 8 | Артефакты-для-отдачи: агент ПРОИЗВОДИТ файл (таблица → .xlsx/.csv) и ДОСТАВЛЯЕТ его пользователю как скачиваемый файл, а не вываливает CSV-текст в чат. |
| [src/runtime/compute.py](../src/runtime/compute.py) | 88 | 6 | Вычислительный слой агента: исполнение Python в песочнице как ИНСТРУМЕНТ. |
| [src/runtime/context_files.py](../src/runtime/context_files.py) | 109 | 6 | Root-convention файлы проекта — как `CLAUDE.md` у Claude Code, только под `sea`. |
| [src/runtime/data_fetch.py](../src/runtime/data_fetch.py) | 132 | 7 | fetch_data — read-only добыча ПОЛНОГО датасета по URL (CSV/CSV.gz/JSON/bulk) → .xlsx-артефакт. |
| [src/runtime/degradation.py](../src/runtime/degradation.py) | 50 | 4 | Видимость тихих деградаций (долг ревью #5). |
| [src/runtime/hitl.py](../src/runtime/hitl.py) | 300 | 20 | Human-in-the-loop: подтверждение side-effect действий перед исполнением — СЕМАНТИЧЕСКОЕ и без спама. |
| [src/runtime/run_context.py](../src/runtime/run_context.py) | 233 | 21 | Единый per-request контекст (run_id + user_id) — изоляция процессно-глобального состояния под КОНКУРЕНТНЫМ сервером (FastAPI `asyncio.create_task`) и Telegram (aiogram). |
| [src/runtime/runbudget.py](../src/runtime/runbudget.py) | 237 | 22 | Токен-бюджет одного прогона — жёсткий предохранитель от runaway. |
| [src/runtime/sea_workspace.py](../src/runtime/sea_workspace.py) | 260 | 10 | `.sea/` — рабочий каталог проекта для CLI-агента `sea` (L1). Как `.git/` у git: per-project локальное состояние. Создаётся `sea init`. Хранит историю/конфиг/лог решений (accept/reject). |
| [src/search/__init__.py](../src/search/__init__.py) | 0 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/search/bandit.py](../src/search/bandit.py) | 52 | 1 | Per-user бандит-прайор выбора режима мышления (контур C, develop.md). |
| [src/search/collective.py](../src/search/collective.py) | 95 | 4 | Коллективный ярус рецептов (контур G) — «рекомендательная система поведения». |
| [src/search/habits.py](../src/search/habits.py) | 70 | 3 | Привычки: само-расширение из ПОВТОРОВ (контур B, develop.md). |
| [src/search/retrieval.py](../src/search/retrieval.py) | 54 | 2 | Канонический лёгкий retrieval для core-модулей (ОДИН механизм над разными корпусами: навыки, память, и т.п. — как в CLAUDE.md). BM25S при наличии, иначе token-overlap. |
| [src/security/__init__.py](../src/security/__init__.py) | 1 | 0 | Deterministic security controls. No LLM decisions or cloud telemetry. |
| [src/security/__main__.py](../src/security/__main__.py) | 26 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/security/attachments.py](../src/security/attachments.py) | 57 | 2 | Bounded attachment admission, not malware detection or a parser sandbox. |
| [src/security/audit.py](../src/security/audit.py) | 133 | 8 | Read-only host inventory plus sandboxed architecture analysis; no LLM/network calls. |
| [src/security/egress.py](../src/security/egress.py) | 85 | 4 | Public HTTP GET with pinned DNS, per-redirect validation, no ambient credentials. |
| [src/security/events.py](../src/security/events.py) | 29 | 1 | Bounded local event metadata; never record prompts, URLs, headers or secrets. |
| [src/security/http.py](../src/security/http.py) | 102 | 9 | Single-owner API boundary: Host/Origin/peer checks and optional token auth. |
| [src/security/integrity.py](../src/security/integrity.py) | 85 | 6 | Operator-enrolled source baseline; model tools cannot enroll or update it. |
| [src/security/sensitive.py](../src/security/sensitive.py) | 25 | 2 | Known-credential egress checks; not a universal personal-data classifier. |
| [src/security/worker.py](../src/security/worker.py) | 46 | 2 | Isolated stdlib-only AST analyst. Reads bounded JSON stdin; executes no project code. |
| [src/skills/app_control/app_control.py](../src/skills/app_control/app_control.py) | 109 | 8 | app_control — СТРУКТУРНОЕ взаимодействие с приложениями (без костылей-скриншотов). |
| [src/skills/ax_control/ax_control.py](../src/skills/ax_control/ax_control.py) | 144 | 10 | ax_control — структурное управление ЛЮБЫМИ приложениями через Accessibility API (AX). |
| [src/skills/browser_control/browser_control.py](../src/skills/browser_control/browser_control.py) | 214 | 16 | browser_control — действия в браузере. ДВА бэкенда, выбор автоматический:   • если подключено браузерное РАСШИРЕНИЕ (агент живёт в ТВОЁМ браузере: твои логины,     твои вкладки, без перелогина) — команды идут туда;   • иначе фолбэк на управ |
| [src/skills/code/code.py](../src/skills/code/code.py) | 263 | 11 | Навык `code` (L2): обзор и правка кодовой базы. Порт паттернов Glob/Grep/FileRead/FileEdit/Bash из ~/Desktop/backup (Claude Code). |
| [src/skills/deep_agent/deep_agent.py](../src/skills/deep_agent/deep_agent.py) | 66 | 1 | deep_agent — ДОПОЛНЕНИЕ к ядру (не замена): запускает LangChain DeepAgent для долгогоризонтных/файловых подзадач. Даёт виртуальную ФС + todo-состояние + спавн суб-агентов «из коробки», не трогая основной граф. |
| [src/skills/device_control/device_control.py](../src/skills/device_control/device_control.py) | 285 | 15 | device_control — действия с устройством по запросу из чата (on-demand, не мониторинг). |
| [src/skills/file_operations/file_operations.py](../src/skills/file_operations/file_operations.py) | 58 | 4 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/file_organizer/file_organizer.py](../src/skills/file_organizer/file_organizer.py) | 40 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/generate_pptx/generate_pptx.py](../src/skills/generate_pptx/generate_pptx.py) | 127 | 4 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/launcher/launcher.py](../src/skills/launcher/launcher.py) | 19 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/link_parser/link_parser.py](../src/skills/link_parser/link_parser.py) | 62 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/orcid_reader/orcid_reader.py](../src/skills/orcid_reader/orcid_reader.py) | 86 | 2 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/phone_control/phone_control.py](../src/skills/phone_control/phone_control.py) | 172 | 10 | phone_control — структурное управление Android-телефоном через ADB (не скриншоты). |
| [src/skills/pubchem_query/pubchem_query.py](../src/skills/pubchem_query/pubchem_query.py) | 71 | 4 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/stash/stash.py](../src/skills/stash/stash.py) | 220 | 15 | stash — именованные рабочие наборы структурированных данных (бюджет, таблицы, трекеры, аналитика). Персональный «рабочий стол» агента под рутину пользователя: фин-аналитик ведёт бюджет, разработчик — список задач, и т.п. |
| [src/skills/text_file_processor/text_file_processor.py](../src/skills/text_file_processor/text_file_processor.py) | 76 | 3 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/uv_package_manager/uv_package_manager.py](../src/skills/uv_package_manager/uv_package_manager.py) | 22 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/weather_check/weather_check.py](../src/skills/weather_check/weather_check.py) | 17 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/web_search/web_search.py](../src/skills/web_search/web_search.py) | 577 | 27 | web_search — поиск и чтение страниц. Произвольные URL читаются через ограниченный HTTP-транспорт с проверкой DNS/redirect; браузер остаётся у фиксированных поисковиков. |
| [src/skills/wikipedia_twitter_citations/wikipedia_twitter_citations.py](../src/skills/wikipedia_twitter_citations/wikipedia_twitter_citations.py) | 132 | 4 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/skills/zip_extractor/zip_extractor.py](../src/skills/zip_extractor/zip_extractor.py) | 33 | 1 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/tools/__init__.py](../src/tools/__init__.py) | 13 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/tools/image_search.py](../src/tools/image_search.py) | 120 | 6 | Поиск картинок для показа В ЧАТЕ (desktop-GUI рендерит markdown ![](url) как картинку). |
| [src/tools/openclaw_import.py](../src/tools/openclaw_import.py) | 269 | 8 | Импорт навыков OpenClaw (ClawHub) в нашу библиотеку навыков. |
| [src/tools/skill_creation.py](../src/tools/skill_creation.py) | 1142 | 52 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/tools/skill_health.py](../src/tools/skill_health.py) | 137 | 10 | Здоровье навыков во времени (контур самопочинки). |
| [src/tools/skill_repair.py](../src/tools/skill_repair.py) | 140 | 5 | Самопочинка навыков (замыкание контура здоровья). |
| [src/tools/skills_import.py](../src/tools/skills_import.py) | 137 | 2 | Импорт ВНЕШНЕГО навыка (папка или .zip от пользователя) с проверкой ПЕРЕД приёмом. |
| [src/tools/utils_validation.py](../src/tools/utils_validation.py) | 174 | 15 | AST-анализ кода СГЕНЕРИРОВАННЫХ навыков (реальный гейт, не строковый матчинг). |
| [src/tracing/__init__.py](../src/tracing/__init__.py) | 4 | 0 | Нет описания модуля; см. исходник и символы в inventory.json. |
| [src/tracing/__main__.py](../src/tracing/__main__.py) | 37 | 1 | python -m src.tracing            # самодиагностика по трейсам за сутки python -m src.tracing --hours 72 |
| [src/tracing/diagnose.py](../src/tracing/diagnose.py) | 63 | 1 | Самодиагностика: агент сам находит свои «косяки» из трейсов и статистики качества. |
| [src/tracing/tracer.py](../src/tracing/tracer.py) | 184 | 15 | Полный трейсинг агента по нодам графа. |
| [src/training/__init__.py](../src/training/__init__.py) | 1 | 0 | Compatibility package. Training implementation lives in src.agent_train. |
| [src/training/verified_edits.py](../src/training/verified_edits.py) | 2 | 0 | Compatibility import; use src.agent_train.verified_edits. |
| [src/utils.py](../src/utils.py) | 428 | 10 | Нет описания модуля; см. исходник и символы в inventory.json. |
