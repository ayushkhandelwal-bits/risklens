# AI Risk Analyst

## Why the AI needs tools (and not just a prompt)
A chatbot on its own invents plausible numbers. A risk analyst needs **evidence**. In RiskLens the language model never sees the database directly. It can only call named, typed tools that run governed queries and model functions. Every number in an answer therefore traces back to a tool call that is recorded in the audit log. The model's job is to plan the investigation and explain the findings; the platform's job is to compute them.

## Components
| File | Role |
|---|---|
| `ai/agent.py` | Orchestrates the investigation: system prompt, tool loop (max 10 steps), error capture, audit, fallback |
| `ai/llm.py` | Provider layer with native tool calling: Anthropic Claude (`claude-sonnet-5-5` default), OpenAI (`gpt-5.5`), or the OpenAI-compatible endpoints of Google Gemini, Groq and xAI (free tiers exist for Gemini and Groq) |
| `ai/planner.py` | Offline mode: deterministic intent routing → same tools → templated answer built only from tool outputs |
| `ai/prompts/system.md` | Rules: tool-grounded numbers, test the premise, materiality, provenance labels, decision support only, output formats |
| `ai/tools/*.py` | 10 tools, registered once and reused by the agent and the MCP server |
| `ai/tools/context.py` | Passes the analyst's dashboard filter selection to tools as default context |
| `mcp_server/server.py` | Same tools over the Model Context Protocol |

## Tools
| Tool | Inputs | Returns |
|---|---|---|
| `get_customer_profile` | customer_id | identity, outcome, application, exposure, payments, utilisation, previous apps, bureau, behavioural flags |
| `calculate_customer_risk` | customer_id | **live re-score** with the champion model, stored PD (consistency check), challenger PD, tier, score, LGD/EAD/EL, portfolio percentile |
| `explain_prediction` | customer_id, top_n | SHAP risk-increasing and decreasing drivers, contribution by source, base PD, reconstruction error |
| `get_early_warning_signals` | customer_id *or* filters | triggers with severity and evidence, or alert counts, top triggers and band validation |
| `get_anomaly_signals` | customer_id *or* filters | rule evidence and isolation percentile, or status counts, rules and most anomalous customers |
| `get_portfolio_metrics` | filters, group_by | KPIs and a breakdown |
| `generate_root_cause_analysis` | metric, comparison (vintage / intake), filters | change, materiality, segment mix vs rate effects, SHAP driver shift, behavioural shifts, EWS/anomaly rate change |
| `get_investigation_priorities` | filters, limit | live-ranked queue with evidence |
| `analyze_model_drift` | top_n | performance, score/feature PSI, segment drift, interpretation |
| `run_safe_sql_query` | sql | rows (max 200) from whitelisted relations, read-only |

## Agent workflow, example 1: "Why is customer C426709 high risk?"
1. `get_customer_profile(C426709)`
2. `calculate_customer_risk(C426709)` gives PD 62.5%, Very High tier, score 99, Expected Loss 75.8K; the live re-score matches the stored score.
3. `explain_prediction(C426709)` gives the drivers: very low external score (0.06), card utilisation 103%, previous approval rate 14%, and others.
4. `get_early_warning_signals(C426709)` gives EWS 100 (Critical), including 6 applications in 90 days, card over limit, 4 refusals in 12 months, and utilisation up 37 points.
5. `get_anomaly_signals(C426709)` gives 92 (Suspicious): application velocity and 4 of 6 address checks mismatching.
6. Synthesis: risk summary → drivers → evidence → signals → recommended analyst actions (limit review, fraud-risk verification, enhanced monitoring), clearly framed as decision support.

## Example 2: "Why has portfolio risk increased?"
`get_portfolio_metrics(group_by=vintage)` → `generate_root_cause_analysis(vintage)` → `generate_root_cause_analysis(intake)` → `get_early_warning_signals()`.

The data says booked-portfolio PD is **stable**: 8.00% → 7.99%, −0.2% relative, below the 3% materiality threshold. The analyst therefore **rejects the premise**, explains the offsetting segment movements (mix vs rate effects) and the SHAP driver shift, and notes that new applicants are about 11% lower-risk on average. It then points to the drifted inputs to check.

## Example 3: "Which customers should the risk team investigate first?"
`get_investigation_priorities()` ranks live data by 30% PD percentile + 25% Expected Loss percentile + 25% EWS + 20% anomaly. The answer is a ranked table with the key reason for each customer. Nothing is hardcoded; change the filters and the list changes.

## Security controls
1. **No raw database access.** Tools only.
2. **SQL guard** (`ai/tools/sql_tool.py`):
   - Single statement, `SELECT`/`WITH` only; comments stripped.
   - Blocks `INSERT UPDATE DELETE DROP ALTER TRUNCATE CREATE GRANT REVOKE COPY … INTO`, `pg_sleep`, `pg_read*`, `dblink`, `current_setting`, `pg_catalog`, `information_schema` and `audit_log`.
   - Only whitelisted relations; CTE names are allowed.
   - Executed as `SELECT * FROM (<q>) LIMIT 201` in a READ ONLY transaction with a 5-second statement timeout.
   - Connects as `risklens_ai`, with SELECT on analytical objects only; tested to refuse DELETE and audit-log reads even if the validator were bypassed.
3. **Audit.** `ai_query` → each `tool_call` (tool, parameters, result summary, status) → `ai_response`, all keyed by session. LLM failures are logged as `ai_error`.
4. **Decision support.** The prompt and planner never approve or reject credit, and anomaly evidence is never called fraud.

## Modes
| Mode | When | Behaviour |
|---|---|---|
| `llm` | `LLM_API_KEY` set (`LLM_PROVIDER` = anthropic, openai, gemini, groq or xai) | the model plans tool calls freely and writes the answer from tool results |
| `planner` | no key, or the LLM fails | deterministic plan for the 7 supported intents; templated answer from tool outputs |
| `*+mcp` | `RISKLENS_TOOL_TRANSPORT=mcp` | every tool call goes through the MCP server (protocol-level) |

**Verification status.** The planner, tools, SQL guard and MCP are tested end-to-end on live data. The LLM loop is tested with the real Anthropic and OpenAI SDK clients driven by mocked API responses (tool_use/tool_result formats, tool-call IDs, the audit trail and fallback). A live run requires your own API key.

## MCP: why it is useful here
MCP turns RiskLens's governed tools into a **standard interface** that any agent can use (Claude Desktop, IDE agents, other internal copilots) without rebuilding integrations, while the same SQL guard, least-privilege role and audit trail still apply. The server exposes 10 tools, 2 resources (`risklens://methodology/assumptions`, `risklens://data/dictionary`) and an `investigate_customer` prompt over stdio or streamable HTTP. Tests cover the in-memory protocol, a real stdio subprocess, and parity with direct calls.
