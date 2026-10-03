# ML Methodology

All figures below come from the build in this repository: 50,000 booked applicants and 12,000 intake applicants, seed 42.

## 1. Target and population
- **Target:** Home Credit `TARGET` (1 = client had payment difficulties), used as a **default proxy**. Observed base rate in the sample: **7.98%** (full dataset: 8.07%).
- **Development population:** booked portfolio only, since intake applicants have no outcome yet.
- **Split:** 60% train / 20% validation / 20% test, stratified on the target, seed 42.
  - Validation is used for early stopping, hyper-parameter choice and the decision threshold.
  - **Test is used once** for every reported metric.

## 2. Leakage controls
- Customer 360 aggregates only behaviour dated before the application (`DAYS_* < 0`, `MONTHS_BALANCE < 0`); `tests/test_etl.py` asserts these guards.
- No target-derived features. IDs are excluded.
- Early-warning and anomaly signals are **not** model inputs.

## 3. Features
52 features in 8 groups (application, applicant, external score, credit bureau, exposure, previous loans, payment behaviour, credit card); see `docs/data_dictionary.md`. Deliberately excluded: gender, family status and children count (fair-lending hygiene).

## 4. Models
| | Champion: XGBoost | Challenger: Logistic Regression |
|---|---|---|
| Preprocessing | none (native missing handling) | winsorise 1–99%, median impute + missing indicators, standardise |
| Key settings | depth 3, learning rate 0.02, min_child_weight 60, subsample 0.8, colsample 0.5, λ = 5, early stopping 200 (best iteration 1,733) | L2, C = 0.05 |
| Class weighting | none, so PDs stay calibrated | none |
| Train AUC | 0.838 | 0.770 |
| **Test AUC** | **0.7490** | 0.7398 |
| Gini | 0.4981 | 0.4795 |
| KS | 0.3761 | 0.3589 |
| Brier (baseline 0.0734) | 0.0676 | 0.0678 |
| Mean PD ÷ observed default | 0.999 | 1.001 |
| Max calibration gap (deciles) | 3.0 pts | 1.3 pts |
| Threshold (F1-optimal on validation) | PD ≥ 0.162 | PD ≥ 0.121 |
| Precision / Recall / F1 (test) | 0.237 / 0.396 / 0.297 | 0.199 / 0.496 / 0.284 |

**Tuning.** A small grid on the *validation* split covered depth {3, 4, 5}, min_child_weight {20, 60} and colsample {0.5, 0.7}. All configurations scored within 0.002 validation AUC of each other, so the least-overfit one was kept.

**Why XGBoost is the champion:** it has the best test discrimination, it captures interactions (for example utilisation × payment behaviour) and missing-history patterns, and it is exactly explainable with TreeSHAP. The logistic challenger remains a transparent benchmark; the small gap (+0.009 AUC) is reported honestly.

**Why accuracy is not used:** a model predicting "no default" for everyone scores 92% accuracy.

## 5. Risk tiers and risk score
| Tier | PD range | Booked customers | Mean PD | Observed default |
|---|---|---|---|---|
| Low | < 5% | 25,918 | 2.5% | 2.0% |
| Medium | 5–10% | 11,351 | 7.1% | 6.9% |
| High | 10–20% | 8,119 | 14.0% | 13.9% |
| Very High | ≥ 20% | 4,612 | 30.5% | 33.8% |

*(Booked portfolio, mixing train, validation and test rows; held-out test-only calibration is shown on the Credit Risk page.)*

**Risk score** = PD percentile within the development (train) population, 0–100, where 100 is the riskiest.

## 6. Explainability (SHAP)
- `shap.TreeExplainer` on the champion booster. Values are in log-odds and additive: `logit(PD) = base + Σ SHAP`. Every explanation reports its reconstruction error (around 1e-7).
- Bulk explanations use XGBoost's native `pred_contribs` (the same TreeSHAP algorithm). It was verified identical to `shap.TreeExplainer` (maximum difference 0.0) before use.
- The booster is sliced to the best iteration at training time, so PD and SHAP describe exactly the same trees.
- Stored: the top 5 risk-increasing and top 5 risk-decreasing drivers per customer, plus global mean |SHAP|.
- Top global drivers: external credit score (average), years in employment, loan-to-goods ratio, external scores 2 and 3, bureau debt-to-credit, share of instalments paid late.

## 7. Expected Loss
`EL = PD × LGD × EAD`
- **PD:** champion model (MODEL).
- **LGD:** ASSUMPTION. 45% for cash loans (Basel foundation-IRB reference for senior unsecured exposures), 65% for revolving.
- **EAD:** DERIVED. Cash loans use the loan amount; revolving uses the limit × CCF 0.75 (an ASSUMPTION).
- Portfolio Expected Loss: 1.00B on 30.1B exposure (3.38% EL rate). Change the assumptions in `common/config.py`.

## 8. Early Warning System
Rule-based and explainable, defined in `sql/early_warning.sql`. Trigger thresholds were chosen by profiling each signal's lift in observed default on the booked book:

| Signal (example threshold) | Share of book | Default when fired | Otherwise |
|---|---|---|---|
| ≥ 2 missed payments in 12m | 0.1% | 37.0% | 8.0% |
| Card utilisation ≥ 100% | 2.2% | 18.9% | 7.7% |
| Late rate in 12m ≥ 30% | 5.1% | 13.0% | 7.7% |
| Bureau account currently overdue | 1.2% | 14.6% | 7.9% |
| ≥ 2 refusals in 12m | 7.3% | 12.9% | 7.6% |
| Credit enquiries in 3m ≥ 2 | 5.5% | 8.1% | 8.0% → **dropped (no lift)** |

Score validation (observed default by band): **Low 7.3% · Medium 13.6% · High 17.6% · Critical 28.8%**.
*Caveat:* thresholds were set in-sample. Production use would recalibrate on an out-of-time period.

## 9. Behavioural anomaly detection
- Isolation Forest (300 trees, contamination 3%) on 20 behavioural and application features (log-transformed amounts, robust-scaled, median-imputed), converted to a percentile.
- 8 transparent triage rules, 40 points each.
- `anomaly_score = 0.6 × isolation percentile + 0.4 × rule score`. Normal ≤ 60, Watch 61–80, Suspicious > 80.
- Result: Normal 90.9%, Watch 8.5%, Suspicious 0.6%. Observed default is 7.8% / 9.4% / 10.2%. Anomaly is deliberately a different signal from credit risk, and it is **not** fraud detection (no labels exist).

## 10. Model monitoring
- **PSI** with baseline-decile bins plus a separate missing bin, ε = 1e-4. Thresholds: 0.10 warning, 0.25 critical.
- Baseline = train split; comparison = recent intake.
- Results:
  - Score PSI 0.010 (stable).
  - Critical: implied loan term (PSI 1.01; mean 21.7 → 17.2 months).
  - Warning: loan-to-income (0.117), loan-to-goods (0.114).
  - Segment mix: revolving share 9.4% → 1.0% (PSI 0.20).
- Performance stability: test-split AUC by vintage (0.71–0.80, about 1,200 customers each, roughly ±0.03 noise) and by product (cash 0.749, revolving 0.742).
- Performance on the intake cannot be measured until outcomes arrive. That is why drift is monitored.

## 11. Known limitations
Simulated vintages (no dates) · default proxy target · LGD/EAD assumptions · no fraud labels · in-sample EWS thresholds · single random split (no out-of-time validation) · sample of 62,000 applicants.
