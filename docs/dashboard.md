# Dashboard Guide

**Design language:** an enterprise risk control tower. A dark navigation rail, a light working canvas, one accent colour (blue) for data, and status colours (green / amber / orange / red) reserved for **status only** and always paired with an icon and label. Thin marks, hairline grids, rate axes that start at zero, no 3D, no rainbow palettes, no gauges. Every KPI card carries a provenance tag:

| Tag | Meaning |
|---|---|
| **OBSERVED** | straight from source data (exposure, default rate, record counts) |
| **MODEL** | champion model outputs (PD, risk tier, Expected Loss) |
| **DERIVED** | rule/ML signals (early-warning score, anomaly score, PSI) |
| **ASSUMPTION** | documented configuration (LGD, CCF) |

Filters live on the **Portfolio** page and are shared by every page and by the AI Risk Analyst. The current selection is shown on each page.

## Pages
### Overview: Portfolio Health
- **KPIs:** Portfolio Exposure · Default Rate · Average PD · Expected Loss · High-Risk Customers · Active Early Warnings.
- **Insight:** generated from live queries (period comparison with materiality, offsetting segments, behavioural shift, SHAP drivers), with a "how this was produced" panel.
- **Charts:** Portfolio Risk Trend by vintage (observed vs model) · Exposure by Risk Tier with EL labels · PD distribution · Early Warning Trends (utilisation, late-payment and POS delinquency rates over the 24 months before application) · Top Risk Drivers.
- **Data-quality strip.**

### Portfolio
- **Seven multi-select filters:** Vintage, Product, Income band, Region, Customer segment, Risk tier, Credit score band.
- **On every change, everything below re-queries the API:** KPIs (Exposure, Default Rate, Average PD, Expected Loss, Customers, High-Risk %), risk by vintage / segment / product (observed vs PD), exposure concentration by PD decile, PD distribution, and the customer table.
- **Selecting a customer row** opens them in the Investigation Center or the AI analyst.

### Credit Risk
- Champion and challenger cards (AUC, Gini, KS, Brier, calibration, precision/recall/F1, overfit check).
- **Tabs:** Performance (ROC, calibration, PD distribution) · Explainability (global SHAP, attribution by data source, driver analysis for the selection) · Expected Loss (by tier and segment, assumptions panel) · Risk by segment.

### Early Warnings
- **KPIs:** total, critical, high and medium alerts, plus exposure under alert.
- **Charts:** alerts by trigger and severity; observed default by band (shows the score carries risk signal).
- **Alert queue** filterable by band and trigger. Selecting a customer shows "Why this customer triggered an alert", listing each trigger with severity and evidence.
- **Behavioural anomalies:** status counts, rule frequency and the not-fraud disclaimer.

### Investigations
- **Customer selection:** search a customer ID, or pick one from the priority queue.
- **Header:** status pills (risk tier, EWS band, anomaly status) and the observed outcome.
- **KPIs:** Risk Score, PD (and challenger), Exposure, Expected Loss (with the LGD × EAD breakdown), EWS, Anomaly.
- **Tabs:** SHAP explanation (waterfall of drivers from the average PD to the customer's PD) · Payment behaviour (monthly on-time vs late) · Credit utilisation (monthly, with the limit line) · Previous applications · Bureau history · Signals · Customer profile.
- **[Investigate with AI]** launches the analyst on this customer. Each opened customer writes an audit event.

### AI Risk Analyst
- Chat with six suggested questions.
- Header pills show the mode (LLM or offline planner), the current filter context and the tool count.
- A live status block lists each tool as it runs ("Retrieving customer profile…", "Calculating customer risk…", "Generating root-cause analysis…"); an expandable trace shows the arguments, durations and result summaries.

### Model Monitor
- **Status pills** and **KPIs:** AUC, Gini, KS, Brier, Calibration, Score PSI.
- **[Explain Model Drift]** runs the AI analyst with the drift tool; **[Re-run monitoring]** recomputes PSI.
- **Charts:** feature PSI (top 15, with warning/critical thresholds) · distribution comparison for any feature · population changes · AUC by vintage and product · interpretation text.

### Data & Audit
- **Data quality:** status, total records, passed/warning/failed checks, completeness, raw-layer row counts, and a check table.
- **Model registry:** versions, roles, training date, features, metrics, hyper-parameters.
- **Audit trail:** filterable by action (ai_query, tool_call, ai_response, investigation_opened, risk_scoring, model_registered…).

## Interactivity guarantees
- The cache key includes every filter value (TTL 30 s), so a changed filter is never answered from stale cache.
- `tests/test_api.py::test_filters_change_results` and `test_filter_result_matches_direct_sql` assert that filtered API results change and match direct SQL.
- Browser-verified: Product = Revolving takes the KPIs from 50,000 customers to 4,711; adding Region Tier 3 takes them to 626 at a 9.27% default rate.
