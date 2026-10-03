-- =====================================================================
-- PORTFOLIO ANALYTICS LAYER
-- v_portfolio is the single analytical surface used by the API: one row per
-- customer, combining observed data (Customer 360) with risk-engine outputs.
-- Risk-engine tables are LEFT JOINed so analytics work before models exist.
-- =====================================================================
DROP VIEW IF EXISTS v_investigation_queue CASCADE;
DROP VIEW IF EXISTS v_portfolio_by_vintage CASCADE;
DROP VIEW IF EXISTS v_portfolio CASCADE;

CREATE VIEW v_portfolio AS
SELECT
    c.sk_id_curr, c.customer_id, c.population, c.default_flag,
    c.vintage, c.product, c.region, c.income_band, c.customer_segment, c.credit_score_band,
    c.age_years, c.amt_income_total, c.amt_credit, c.amt_annuity,
    c.credit_to_income, c.annuity_to_income, c.ext_source_mean,
    c.current_exposure, c.total_exposure,
    c.cc_util_avg, c.cc_util_3m, c.cc_util_change,
    c.inst_late_rate, c.inst_late_rate_12m, c.inst_late_rate_change, c.inst_missed_12m,
    c.prev_apps_365d, c.prev_refused_365d, c.bureau_enquiries_12m,
    c.bureau_active, c.bureau_delinquent, c.bureau_debt_total,
    -- model outputs (NULL until the risk engine has scored)
    r.pd, r.pd_challenger, r.risk_score, r.risk_tier, r.lgd, r.ead, r.expected_loss, r.dataset_split,
    r.model_version,
    -- early warning
    e.ews_score, e.ews_band, e.n_triggers AS ews_triggers, e.top_trigger AS ews_top_trigger,
    -- behavioural anomaly
    a.anomaly_score, a.anomaly_status, a.n_rules AS anomaly_rules
FROM customer_360 c
LEFT JOIN risk_scores r           USING (sk_id_curr)
LEFT JOIN early_warning_signals e USING (sk_id_curr)
LEFT JOIN anomaly_scores a        USING (sk_id_curr);

-- Vintage roll-up (booked portfolio). Default rate is OBSERVED; PD/EL are MODEL outputs.
CREATE VIEW v_portfolio_by_vintage AS
SELECT vintage,
       COUNT(*)                                        AS customers,
       SUM(current_exposure)                           AS exposure,
       AVG(default_flag::float)                        AS default_rate,
       AVG(pd)                                         AS avg_pd,
       SUM(expected_loss)                              AS expected_loss,
       AVG((risk_tier IN ('High','Very High'))::int)   AS high_risk_share,
       AVG((ews_score >= 31)::int)                     AS ews_alert_rate,
       AVG(cc_util_3m)                                 AS avg_recent_utilisation,
       AVG(inst_late_rate_12m)                         AS avg_recent_late_rate
FROM v_portfolio
WHERE population = 'portfolio'
GROUP BY vintage;

-- Behavioural trend over observation months (REAL relative time in the source:
-- MONTHS_BALANCE is months before the application date). Used for "Early Warning Trends".
DROP MATERIALIZED VIEW IF EXISTS mv_behaviour_trend;
CREATE MATERIALIZED VIEW mv_behaviour_trend AS
WITH pos AS (
    SELECT months_balance AS month, COUNT(DISTINCT sk_id_curr) AS active,
           COUNT(DISTINCT sk_id_curr) FILTER (WHERE sk_dpd > 0) AS delinquent
    FROM raw_pos_cash WHERE months_balance BETWEEN -24 AND -1 GROUP BY 1
), cc AS (
    SELECT months_balance AS month,
           AVG(amt_balance / NULLIF(amt_credit_limit_actual,0)) AS utilisation,
           COUNT(DISTINCT sk_id_curr) FILTER (WHERE sk_dpd > 0)::float / NULLIF(COUNT(DISTINCT sk_id_curr),0) AS cc_dpd_rate
    FROM raw_credit_card WHERE months_balance BETWEEN -24 AND -1 AND amt_credit_limit_actual > 0 GROUP BY 1
), inst AS (
    SELECT (FLOOR(days_instalment / 30.44))::int AS month,
           AVG((days_entry_payment > days_instalment)::int) AS late_rate
    FROM raw_installments WHERE days_instalment BETWEEN -730 AND -1 GROUP BY 1
)
SELECT pos.month,
       pos.delinquent::float / NULLIF(pos.active,0) AS pos_dpd_rate,
       cc.utilisation, cc.cc_dpd_rate, inst.late_rate
FROM pos LEFT JOIN cc USING (month) LEFT JOIN inst USING (month)
ORDER BY pos.month;

-- Analyst work queue: ranked by a transparent priority score (weights in config).
CREATE VIEW v_investigation_queue AS
WITH ranked AS (
    SELECT p.*,
           PERCENT_RANK() OVER (ORDER BY pd)            AS pd_pct,
           PERCENT_RANK() OVER (ORDER BY expected_loss) AS el_pct
    FROM v_portfolio p
    WHERE pd IS NOT NULL
)
SELECT customer_id, sk_id_curr, population, vintage, product, customer_segment,
       pd, risk_score, risk_tier, current_exposure, expected_loss,
       ews_score, ews_band, ews_top_trigger, anomaly_score, anomaly_status,
       ROUND((100 * (0.30 * pd_pct + 0.25 * el_pct
                     + 0.25 * COALESCE(ews_score,0) / 100.0
                     + 0.20 * COALESCE(anomaly_score,0) / 100.0))::numeric, 1)::float AS priority_score
FROM ranked;
