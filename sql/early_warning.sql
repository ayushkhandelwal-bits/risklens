-- =====================================================================
-- EARLY WARNING SYSTEM — transparent rule book
-- Each trigger fires from Customer 360 behaviour and carries a severity and
-- the evidence that fired it. Severities were set from each signal's observed
-- lift in default rate on the booked portfolio (see docs/ml_methodology.md).
--   CRITICAL = 35 pts   HIGH = 20 pts   MEDIUM = 10 pts   (score capped at 100)
-- Bands: 0-30 Low · 31-60 Medium · 61-80 High · 81-100 Critical
-- =====================================================================
TRUNCATE ews_triggers;

INSERT INTO ews_triggers (sk_id_curr, trigger_code, severity, description, evidence)
-- 1. Payment behaviour deteriorated (last 12m vs the 12m before)
SELECT sk_id_curr, 'PAYMENT_DETERIORATION',
       CASE WHEN inst_late_rate_12m >= 0.30 AND inst_late_rate_change >= 0.10 THEN 'CRITICAL'
            WHEN inst_late_rate_change >= 0.10 AND inst_late_rate_12m >= 0.15 THEN 'HIGH'
            ELSE 'MEDIUM' END,
       'Payment behaviour deteriorated',
       'Late-payment rate ' || ROUND((inst_late_rate_12m*100)::numeric,1) || '% in last 12m vs '
         || COALESCE(ROUND((inst_late_rate_prior*100)::numeric,1)::text, 'n/a') || '% in the prior 12m'
FROM customer_360
WHERE (inst_late_rate_change >= 0.10 AND inst_late_rate_12m >= 0.15) OR inst_late_rate_12m >= 0.30

UNION ALL -- 2. Missed / under-paid instalments
SELECT sk_id_curr, 'MISSED_PAYMENTS',
       CASE WHEN inst_missed_12m >= 2 THEN 'CRITICAL' ELSE 'HIGH' END,
       'Recent missed payments',
       inst_missed_12m || ' instalment(s) paid less than 95% of amount due in the last 12m'
FROM customer_360 WHERE inst_missed_12m >= 1

UNION ALL -- 3. Card utilisation at / over limit
SELECT sk_id_curr, 'HIGH_UTILISATION',
       CASE WHEN cc_util_3m >= 1.0 THEN 'HIGH' ELSE 'MEDIUM' END,
       CASE WHEN cc_util_3m >= 1.0 THEN 'Credit card over limit' ELSE 'Credit card near limit' END,
       'Card utilisation ' || ROUND((cc_util_3m*100)::numeric,0) || '% over the last 3 months'
FROM customer_360 WHERE cc_util_3m >= 0.90

UNION ALL -- 4. Utilisation increasing
SELECT sk_id_curr, 'UTILISATION_INCREASE',
       CASE WHEN cc_util_change >= 0.30 THEN 'HIGH' ELSE 'MEDIUM' END,
       'Credit utilisation increased',
       'Card utilisation up ' || ROUND((cc_util_change*100)::numeric,0) || ' pts (from '
         || ROUND((cc_util_prior*100)::numeric,0) || '% to ' || ROUND((cc_util_3m*100)::numeric,0) || '%)'
FROM customer_360 WHERE cc_util_change >= 0.15

UNION ALL -- 5. Outstanding balance increasing
SELECT sk_id_curr, 'BALANCE_INCREASE', 'MEDIUM',
       'Outstanding balance increasing',
       'Average card balance ' || TO_CHAR(cc_balance_3m, 'FM999,999,990') || ' (last 3m) vs '
         || TO_CHAR(cc_balance_prior, 'FM999,999,990') || ' (prior 9m)'
FROM customer_360 WHERE cc_balance_3m >= 1.5 * cc_balance_prior AND cc_balance_3m > 50000

UNION ALL -- 6. Credit seeking: application frequency
SELECT sk_id_curr, 'APPLICATION_FREQUENCY',
       CASE WHEN prev_apps_90d >= 3 THEN 'HIGH' ELSE 'MEDIUM' END,
       'Recent application frequency increased',
       prev_apps_90d || ' application(s) in the last 90 days, ' || prev_apps_365d || ' in the last 12 months'
FROM customer_360 WHERE prev_apps_90d >= 2 OR prev_apps_365d >= 5

UNION ALL -- 7. Recent refusals
SELECT sk_id_curr, 'RECENT_REFUSALS',
       CASE WHEN prev_refused_365d >= 2 THEN 'HIGH' ELSE 'MEDIUM' END,
       'Recently refused by lenders',
       prev_refused_365d || ' refused application(s) in the last 12 months'
FROM customer_360 WHERE prev_refused_365d >= 1

UNION ALL -- 8. Bureau behaviour deteriorating
SELECT sk_id_curr, 'BUREAU_DETERIORATION',
       CASE WHEN bureau_delinquent > 0 AND bureau_dpd_months_12m >= 2 THEN 'CRITICAL' ELSE 'HIGH' END,
       'Deteriorating bureau behaviour',
       bureau_dpd_months_12m || ' month(s) past due at other lenders in the last 12m (prior 12m: '
         || bureau_dpd_months_prior || '); ' || bureau_delinquent || ' account(s) currently overdue'
FROM customer_360
WHERE bureau_delinquent > 0 OR (bureau_dpd_months_12m > bureau_dpd_months_prior AND bureau_dpd_months_12m >= 2)

UNION ALL -- 9. Exposure increasing: new credit lines
SELECT sk_id_curr, 'EXPOSURE_INCREASE', 'MEDIUM',
       'Credit exposure increasing',
       bureau_opened_12m || ' new bureau account(s) opened in the last 12m vs ' || bureau_opened_prior_12m
         || ' in the prior 12m; total debt-to-income ' || ROUND(total_debt_to_income::numeric,1)
FROM customer_360 WHERE bureau_opened_12m >= 2 AND bureau_opened_12m > bureau_opened_prior_12m

UNION ALL -- 10. POS / cash loan delinquency
SELECT sk_id_curr, 'POS_DELINQUENCY',
       CASE WHEN pos_dpd_months_12m >= 3 THEN 'HIGH' ELSE 'MEDIUM' END,
       'POS / cash loan delinquency',
       pos_dpd_months_12m || ' month(s) past due on POS/cash loans in the last 12m (max DPD '
         || COALESCE(pos_max_dpd, 0) || ')'
FROM customer_360 WHERE pos_dpd_months_12m >= 1;

-- ------------------------------------------------------------------ score
TRUNCATE early_warning_signals;
INSERT INTO early_warning_signals (sk_id_curr, ews_score, ews_band, n_triggers, top_trigger, top_severity)
SELECT c.sk_id_curr,
       LEAST(100, COALESCE(t.points, 0))                                   AS ews_score,
       CASE WHEN COALESCE(t.points,0) <= 30 THEN 'Low'
            WHEN t.points <= 60 THEN 'Medium'
            WHEN t.points <= 80 THEN 'High' ELSE 'Critical' END            AS ews_band,
       COALESCE(t.n, 0), t.top_trigger, t.top_severity
FROM customer_360 c
LEFT JOIN (
    SELECT sk_id_curr,
           SUM(CASE severity WHEN 'CRITICAL' THEN 35 WHEN 'HIGH' THEN 20 ELSE 10 END) AS points,
           COUNT(*) AS n,
           (ARRAY_AGG(description ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 ELSE 3 END))[1] AS top_trigger,
           (ARRAY_AGG(severity    ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 ELSE 3 END))[1] AS top_severity
    FROM ews_triggers GROUP BY sk_id_curr
) t USING (sk_id_curr);
ANALYZE early_warning_signals;
ANALYZE ews_triggers;
