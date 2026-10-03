# Business Use Cases, Demo Script and Interview Guide

## Use cases by user
| User | Question they bring | Where RiskLens answers it |
|---|---|---|
| Risk Manager | "How healthy is the book? Is risk rising? Where is the loss concentrated?" | Overview KPIs and insight · Portfolio concentration · AI RCA |
| Credit Risk Analyst | "Why is this customer risky? What changed in their behaviour?" | Investigation Center (SHAP, timelines) · AI customer investigation |
| Fraud Risk Analyst | "Which behaviour looks unusual and needs verification?" | Early Warnings → anomalies · investigation queue |
| Collections / portfolio monitoring | "Who is deteriorating before they default?" | Early Warning System with trigger evidence |
| Model Risk team | "Is the model still valid on today's applicants?" | Model Monitor (PSI, segment drift, performance stability) · registry |
| Data Analyst | "Can I trust the data? Can I slice it fast?" | Data & Audit · Portfolio filters · API · read-only SQL tool |

## Decisions it supports (the human stays in charge)
Prioritising review queues · limit reviews on deteriorating revolving lines · verifying identity / income on anomalous applications · enhanced monitoring of Critical early warnings · deciding whether a model needs recalibration when inputs drift · sizing provisions with Expected Loss under stated assumptions.

**Scenario analysis, not achieved results.** If the riskiest PD decile, which holds a disproportionate share of Expected Loss (see the Portfolio concentration chart), received earlier intervention, the avoidable loss could be sized as *EL of that decile × assumed intervention effectiveness*. Any such number is a **scenario** under an assumed effectiveness, not a measured outcome.

## 5-minute demo talk track
1. **Overview (45s).** "This is the risk control tower for a digital lender's 30.1B book: observed default 7.98%, model PD 8.00% (well calibrated), Expected Loss 1.0B. The insight is generated from live queries, and it says risk is stable but segments are offsetting."
2. **Portfolio (45s).** Select *Risk tier = High, Very High*. "Every number recalculates in Postgres: customers, exposure, EL, concentration. No static charts." Show that the riskiest decile carries a large share of EL.
3. **Customer (60s).** Open a top-priority customer. "One workspace: Customer 360 from six source systems, 62.5% PD, Very High tier. SHAP shows why: a very low external score and card utilisation over 100%. The EWS shows the behaviour: 6 applications in 90 days and 4 refusals. The anomaly flags are for review, not a fraud verdict."
4. **AI (60s).** Click *Investigate with AI* and expand the trace. "Five governed tool calls on live data, each audited. The recommendation is decision support." Then ask *"Why has portfolio risk increased?"*: "It tests the premise. Booked risk hasn't materially increased (−0.2%). It decomposes segments into mix vs rate and suggests where to look."
5. **Model Monitor (45s).** "AUC 0.749 on held-out data, score PSI stable, but implied loan term is in critical drift: new applicants take shorter loans, and the revolving share collapsed from 9% to 1%." Click *Explain Model Drift*.
6. **Close (15s).** "Data quality → SQL Customer 360 → calibrated, explainable models → early warning and anomaly signals → governed AI and MCP → one product, with tests and an audit trail."

## Interview guide: be ready to explain
**Why does the business need this?** Risk work today is fragmented across SQL, spreadsheets and notebooks. RiskLens consolidates monitoring, investigation, explanation and model governance, and it turns signals into prioritised actions.

**Why these data sources?** Each represents a system a lender actually runs:
- Origination: affordability.
- Bureau: external indebtedness and delinquency.
- Previous applications: credit-seeking and refusals.
- Payments: repayment discipline.
- Cards: utilisation stress.
- POS: delinquency elsewhere in the relationship.

Risk is multi-source, so the Customer 360 joins all of them.

**Why these features?** They map to classic credit-risk drivers: capacity (loan-to-income, instalment-to-income), character (late rate, missed payments, bureau delinquency), credit-seeking (applications, refusals, enquiries), utilisation stress and trend features (12-month vs prior). They are explainable to a credit committee.

**Why XGBoost?** It has the best held-out discrimination, handles missing histories natively, captures interactions and is exactly explainable with TreeSHAP. Logistic regression is kept as a transparent challenger. Be honest that the gain is modest (+0.009 AUC), which is typical on tabular credit data.

**Why SHAP?** Regulators and analysts need reasons. SHAP decomposes each PD into additive contributions that sum exactly to the model output, so explanations are faithful (verified per customer) and comparable across customers and segments.

**How does Early Warning work?** It uses transparent behavioural rules (payment deterioration, utilisation stress, credit-seeking, bureau deterioration, exposure growth), each with a severity and evidence, scored 0–100. It is not trained on default, yet observed default rises from 7.3% to 28.8% across bands, which validates it.

**How does anomaly detection work?** An Isolation Forest finds profiles that are easy to isolate, meaning unusual; rules add interpretable red flags. Because there are no fraud labels, the output is "review", never "fraud".

**How does Expected Loss work?** EL = PD × LGD × EAD. PD comes from the model; LGD and EAD are documented assumptions because the data has no recovery information.

**How does model monitoring work?** PSI compares each input's and the score's distribution between the training population and new applicants. A stable score with drifting inputs (here, loan term and product mix) means: watch early outcomes and review the pipeline or policy before trusting PDs fully.

**Why does the AI agent need tools?** So every number is computed by governed code on live data instead of generated text. Tools enforce access control (read-only SQL, least privilege), produce an audit trail, and let the model focus on planning and explanation.

**Why is MCP useful?** It exposes the same governed tools through an open standard, so any agent (Claude Desktop, IDE copilots, other teams' assistants) can use RiskLens safely without new integrations. RiskLens's own analyst can route through it too.

**How does the dashboard support analysts?** It goes from book to segment to customer to evidence to action in a few clicks. Filters are shared across pages and with the AI, and the investigation queue tells them where to start.

**What are the dataset's limitations?** Simulated vintages (no dates), a default-proxy target, no recovery data (assumed LGD/EAD), no fraud labels, no outcomes for the intake, historical and anonymised data, and in-sample EWS thresholds. Name these before an interviewer does.
