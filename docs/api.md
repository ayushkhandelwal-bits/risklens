# API Reference

Base URL `http://localhost:8000`. Interactive OpenAPI docs: `/docs`. All responses are JSON.

## Common filter parameters
Every `/portfolio/*`, `/risk/*`, `/early-warnings/*`, `/anomalies/*` and `/investigations/*` endpoint accepts:

| Param | Type | Example |
|---|---|---|
| `population` | `portfolio` (default) · `intake` · `all` | `population=intake` |
| `vintage` | repeatable | `vintage=2025-Q1&vintage=2025-Q2` |
| `product` | repeatable | `product=Revolving loans` |
| `income_band`, `region`, `customer_segment`, `risk_tier`, `credit_score_band`, `ews_band`, `anomaly_status` | repeatable | `risk_tier=High&risk_tier=Very High` |

Values are validated against the data domain (`GET /portfolio/filters`) and bound as SQL parameters.

## Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | API and database status |
| GET | `/portfolio/filters` | valid values per filter dimension |
| GET | `/portfolio/summary` | KPIs: customers, exposure, observed default rate, avg PD, Expected Loss, EL rate, high-risk count/share/exposure, early warnings, anomalies |
| GET | `/portfolio/trend` | metrics by vintage |
| GET | `/portfolio/risk-distribution` | PD histogram + tier totals |
| GET | `/portfolio/breakdown/{dimension}` | metrics by any filter dimension |
| GET | `/portfolio/customers` | customer list (`sort` = expected_loss, pd, exposure, ews, anomaly; `limit`, `offset`, `search`) |
| GET | `/portfolio/concentration` | exposure / EL share by PD decile |
| GET | `/portfolio/behaviour-trend` | monthly utilisation, late-payment and POS delinquency rates (24 months before application) |
| GET | `/portfolio/top-drivers` | most frequent SHAP risk drivers in the selection |
| GET | `/portfolio/insight` | data-generated narrative + evidence |
| GET | `/portfolio/data-quality` | latest data-quality run, checks, coverage |
| GET | `/customers/{id}` | Customer 360 (`?log=true` writes an `investigation_opened` audit event) |
| GET | `/customers/{id}/history` | previous applications, bureau accounts, monthly payments and utilisation |
| GET | `/customers/{id}/risk` | PD, challenger PD, score, tier, LGD/EAD/EL with formula and provenance |
| GET | `/customers/{id}/explanation` | SHAP explanation (`top_n`) |
| GET | `/customers/{id}/warnings` | Early Warning Score + triggers with evidence |
| GET | `/customers/{id}/anomaly` | anomaly score, status, rule evidence |
| GET | `/risk/assumptions` | EL formula, LGD/EAD assumptions, tier and score definitions |
| GET | `/risk/expected-loss/{dimension}` | EL and EL rate by dimension |
| GET | `/risk/drivers` | driver frequency for the selection |
| GET | `/early-warnings/summary` | alerts by band, triggers with observed default rates |
| GET | `/early-warnings/alerts` | alert queue (`band`, `trigger`, `limit`) |
| GET | `/early-warnings/validation` | observed default by EWS band and anomaly status |
| GET | `/anomalies/summary` | anomaly status counts and rule frequency |
| GET | `/investigations/queue` | priority-ranked queue + method |
| GET | `/model/metrics` | registry: champion/challenger metrics, ROC and calibration curves |
| GET | `/model/importance` | global mean absolute SHAP |
| GET | `/model/drift` | monitoring report (`top_n`, `include_bins`) |
| GET | `/model/drift/features` | PSI for every feature |
| GET | `/model/performance/segments` | AUC by vintage and product (test split) |
| POST | `/model/monitoring/run` | recompute monitoring |
| GET | `/ai/status` | LLM configured? provider/model, tool list |
| GET | `/ai/tools` | tool names, descriptions and JSON schemas |
| POST | `/ai/query` | `{question, filters?, session_id?, history?}` → `{answer, mode, model, tool_calls[], notice, elapsed_ms}` |
| POST | `/ai/investigate` | `{customer_id}` → full customer investigation |
| GET | `/audit/log` | audit events (`limit`, `action`, `session_id`) |

## Errors
| Status | `error` | When |
|---|---|---|
| 400 | `bad_request` | unknown breakdown dimension, bad argument |
| 404 | `customer_not_found` | unknown or malformed customer ID |
| 422 | `invalid_filter` / validation | filter value outside the domain, empty AI question |
| 503 | `database_unavailable` | PostgreSQL unreachable |
| 503 | `model_unavailable` | model artefacts missing |

## Examples
```bash
curl "localhost:8000/portfolio/summary?risk_tier=High&risk_tier=Very%20High&product=Revolving%20loans"
curl "localhost:8000/customers/C426709/explanation?top_n=5"
curl -X POST localhost:8000/ai/query -H "content-type: application/json" \
     -d '{"question":"Which accounts require investigation?","filters":{"region":["Region Tier 3"]}}'
```
