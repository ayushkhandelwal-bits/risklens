-- =====================================================================
-- CUSTOMER 360
-- One row per applicant consolidating every source system.
-- Leakage guard: only behaviour dated BEFORE the current application is
-- used (DAYS_* < 0, MONTHS_BALANCE < 0 relative to the application date).
-- =====================================================================
DROP TABLE IF EXISTS customer_360 CASCADE;

CREATE TABLE customer_360 AS
WITH
-- ---------------------------------------------------------------- Credit Bureau
bureau_agg AS (
    SELECT sk_id_curr,
           COUNT(*)                                                         AS bureau_accounts,
           COUNT(*) FILTER (WHERE credit_active = 'Active')                 AS bureau_active,
           COUNT(*) FILTER (WHERE credit_active = 'Closed')                 AS bureau_closed,
           COUNT(*) FILTER (WHERE credit_day_overdue > 0
                               OR COALESCE(amt_credit_sum_overdue,0) > 0)   AS bureau_delinquent,
           SUM(amt_credit_sum_debt) FILTER (WHERE credit_active = 'Active') AS bureau_debt_total,
           SUM(amt_credit_sum)      FILTER (WHERE credit_active = 'Active') AS bureau_credit_active_total,
           MAX(amt_credit_max_overdue)                                      AS bureau_max_overdue_amt,
           -MIN(days_credit) / 365.25                                       AS bureau_history_years,
           COUNT(*) FILTER (WHERE days_credit >= -365)                      AS bureau_opened_12m,
           COUNT(*) FILTER (WHERE days_credit BETWEEN -730 AND -366)        AS bureau_opened_prior_12m
    FROM raw_bureau
    WHERE days_credit < 0
    GROUP BY sk_id_curr
),
bureau_bal AS (
    SELECT b.sk_id_curr,
           COUNT(*) FILTER (WHERE bb.months_balance >= -12 AND bb.status IN ('1','2','3','4','5'))        AS bb_dpd_months_12m,
           COUNT(*) FILTER (WHERE bb.months_balance BETWEEN -24 AND -13 AND bb.status IN ('1','2','3','4','5')) AS bb_dpd_months_prior,
           COUNT(*) FILTER (WHERE bb.status IN ('3','4','5'))                                               AS bb_severe_dpd_months
    FROM raw_bureau_balance bb
    JOIN raw_bureau b USING (sk_id_bureau)
    WHERE bb.months_balance < 0
    GROUP BY b.sk_id_curr
),
-- ---------------------------------------------------------------- Previous Loan System
prev_agg AS (
    SELECT sk_id_curr,
           COUNT(*)                                                             AS prev_app_count,
           COUNT(*) FILTER (WHERE name_contract_status = 'Approved')            AS prev_approved,
           COUNT(*) FILTER (WHERE name_contract_status = 'Refused')             AS prev_refused,
           COUNT(*) FILTER (WHERE days_decision >= -365)                        AS prev_apps_365d,
           COUNT(*) FILTER (WHERE days_decision >= -90)                         AS prev_apps_90d,
           COUNT(*) FILTER (WHERE days_decision >= -365 AND name_contract_status = 'Refused') AS prev_refused_365d,
           AVG(amt_credit) FILTER (WHERE name_contract_status = 'Approved')     AS prev_avg_credit,
           -MAX(days_decision)                                                  AS days_since_last_application
    FROM raw_previous_application
    WHERE days_decision < 0
    GROUP BY sk_id_curr
),
-- ---------------------------------------------------------------- Payment System
-- Payments can be split over several rows: collapse to one row per instalment first.
inst_level AS (
    SELECT sk_id_curr, sk_id_prev, num_instalment_version, num_instalment_number,
           MAX(amt_instalment)       AS amt_due,
           SUM(amt_payment)          AS amt_paid,
           MAX(days_instalment)      AS day_due,
           MAX(days_entry_payment)   AS day_paid
    FROM raw_installments
    WHERE days_instalment < 0
    GROUP BY sk_id_curr, sk_id_prev, num_instalment_version, num_instalment_number
),
inst_agg AS (
    SELECT sk_id_curr,
           COUNT(*)                                                              AS inst_count,
           AVG((day_paid > day_due)::int)                                        AS inst_late_rate,
           AVG(GREATEST(day_paid - day_due, 0))                                  AS inst_avg_days_late,
           MAX(GREATEST(day_paid - day_due, 0))                                  AS inst_max_days_late,
           SUM(CASE WHEN COALESCE(amt_paid,0) < 0.95 * amt_due THEN 1 ELSE 0 END) AS inst_missed_count,
           SUM(amt_paid) / NULLIF(SUM(amt_due), 0)                               AS inst_payment_ratio,
           AVG((day_paid > day_due)::int) FILTER (WHERE day_due >= -365)         AS inst_late_rate_12m,
           AVG((day_paid > day_due)::int) FILTER (WHERE day_due BETWEEN -730 AND -366) AS inst_late_rate_prior,
           SUM(CASE WHEN day_due >= -365 AND COALESCE(amt_paid,0) < 0.95 * amt_due THEN 1 ELSE 0 END) AS inst_missed_12m
    FROM inst_level
    GROUP BY sk_id_curr
),
-- ---------------------------------------------------------------- POS / Cash Loan System
pos_agg AS (
    SELECT sk_id_curr,
           COUNT(*)                                                 AS pos_months,
           COUNT(*) FILTER (WHERE sk_dpd > 0)                       AS pos_dpd_months,
           MAX(sk_dpd)                                              AS pos_max_dpd,
           COUNT(*) FILTER (WHERE sk_dpd > 0 AND months_balance >= -12) AS pos_dpd_months_12m
    FROM raw_pos_cash
    WHERE months_balance < 0
    GROUP BY sk_id_curr
),
-- ---------------------------------------------------------------- Credit Card System
cc_month AS (
    SELECT sk_id_curr, months_balance,
           SUM(amt_balance)                                         AS bal,
           SUM(amt_credit_limit_actual)                             AS lim,
           MAX(sk_dpd)                                              AS dpd
    FROM raw_credit_card
    WHERE months_balance < 0
    GROUP BY sk_id_curr, months_balance
),
cc_agg AS (
    SELECT sk_id_curr,
           AVG(bal / NULLIF(lim, 0))                                         AS cc_util_avg,
           MAX(bal / NULLIF(lim, 0))                                         AS cc_util_max,
           AVG(bal / NULLIF(lim, 0)) FILTER (WHERE months_balance >= -3)     AS cc_util_3m,
           AVG(bal / NULLIF(lim, 0)) FILTER (WHERE months_balance BETWEEN -12 AND -4) AS cc_util_prior,
           AVG(bal) FILTER (WHERE months_balance >= -3)                      AS cc_balance_3m,
           AVG(bal) FILTER (WHERE months_balance BETWEEN -12 AND -4)         AS cc_balance_prior,
           COUNT(*) FILTER (WHERE dpd > 0)                                   AS cc_dpd_months
    FROM cc_month
    GROUP BY sk_id_curr
),
-- ---------------------------------------------------------------- Loan Origination System
app AS (
    SELECT a.*,
           (COALESCE(ext_source_1,0)+COALESCE(ext_source_2,0)+COALESCE(ext_source_3,0))
             / NULLIF((ext_source_1 IS NOT NULL)::int + (ext_source_2 IS NOT NULL)::int + (ext_source_3 IS NOT NULL)::int, 0)
             AS ext_source_mean,
           NTILE(8) OVER (PARTITION BY population ORDER BY sk_id_curr) AS vintage_idx
    FROM raw_application a
)
SELECT
    -- identity & population
    a.sk_id_curr, a.customer_id, a.population, a.target AS default_flag,
    -- business dimensions (vintage is a documented SIMULATION — no dates in source)
    CASE WHEN a.population = 'intake' THEN '2026-Q1'
         ELSE (ARRAY['2024-Q1','2024-Q2','2024-Q3','2024-Q4','2025-Q1','2025-Q2','2025-Q3','2025-Q4'])[a.vintage_idx]
    END                                                             AS vintage,
    a.name_contract_type                                            AS product,
    'Region Tier ' || a.region_rating_client_w_city                 AS region,
    CASE WHEN a.amt_income_total < 100000 THEN '<100K'
         WHEN a.amt_income_total < 150000 THEN '100K–150K'
         WHEN a.amt_income_total < 225000 THEN '150K–225K'
         WHEN a.amt_income_total < 300000 THEN '225K–300K'
         ELSE '300K+' END                                        AS income_band,
    CASE WHEN a.name_income_type IN ('Working','Commercial associate','Pensioner','State servant')
         THEN a.name_income_type ELSE 'Other' END                   AS customer_segment,
    CASE WHEN a.ext_source_mean IS NULL THEN 'Unscored'
         WHEN a.ext_source_mean >= 0.65 THEN 'A · Excellent'
         WHEN a.ext_source_mean >= 0.55 THEN 'B · Good'
         WHEN a.ext_source_mean >= 0.45 THEN 'C · Fair'
         WHEN a.ext_source_mean >= 0.35 THEN 'D · Weak'
         ELSE 'E · Poor' END                                               AS credit_score_band,
    -- demographics / application
    a.code_gender AS gender, a.name_education_type AS education, a.name_family_status AS family_status,
    a.occupation_type AS occupation, a.name_housing_type AS housing,
    ROUND((-a.days_birth / 365.25)::numeric, 1)::float              AS age_years,
    (-a.days_employed / 365.25)                                     AS employment_years,
    a.days_employed_anomaly                                         AS not_employed_flag,
    a.cnt_children, a.cnt_fam_members,
    a.flag_own_car, a.flag_own_realty,
    a.region_rating_client_w_city                                   AS region_rating,
    a.amt_income_total, a.amt_credit, a.amt_annuity, a.amt_goods_price,
    a.amt_credit / NULLIF(a.amt_income_total, 0)                    AS credit_to_income,
    a.amt_annuity / NULLIF(a.amt_income_total, 0)                   AS annuity_to_income,
    a.amt_credit / NULLIF(a.amt_goods_price, 0)                     AS credit_to_goods,
    a.amt_credit / NULLIF(a.amt_annuity, 0)                         AS loan_term_months,
    a.ext_source_1, a.ext_source_2, a.ext_source_3, a.ext_source_mean,
    -a.days_id_publish / 365.25                                     AS id_document_age_years,
    -a.days_last_phone_change                                       AS days_since_phone_change,
    a.amt_req_credit_bureau_mon                                     AS bureau_enquiries_1m,
    a.amt_req_credit_bureau_qrt                                     AS bureau_enquiries_3m,
    a.amt_req_credit_bureau_year                                    AS bureau_enquiries_12m,
    (a.reg_region_not_live_region + a.reg_region_not_work_region + a.live_region_not_work_region
     + a.reg_city_not_live_city + a.reg_city_not_work_city + a.live_city_not_work_city) AS address_mismatch_flags,
    a.def_30_cnt_social_circle                                      AS social_circle_defaults_30d,
    -- bureau
    COALESCE(b.bureau_accounts,0) AS bureau_accounts, COALESCE(b.bureau_active,0) AS bureau_active,
    COALESCE(b.bureau_closed,0)   AS bureau_closed,   COALESCE(b.bureau_delinquent,0) AS bureau_delinquent,
    COALESCE(b.bureau_debt_total,0) AS bureau_debt_total,
    b.bureau_debt_total / NULLIF(b.bureau_credit_active_total, 0)   AS bureau_debt_to_credit,
    b.bureau_max_overdue_amt, b.bureau_history_years,
    COALESCE(b.bureau_opened_12m,0) AS bureau_opened_12m, COALESCE(b.bureau_opened_prior_12m,0) AS bureau_opened_prior_12m,
    COALESCE(bb.bb_dpd_months_12m,0) AS bureau_dpd_months_12m, COALESCE(bb.bb_dpd_months_prior,0) AS bureau_dpd_months_prior,
    COALESCE(bb.bb_severe_dpd_months,0) AS bureau_severe_dpd_months,
    -- previous applications
    COALESCE(p.prev_app_count,0) AS prev_app_count, COALESCE(p.prev_approved,0) AS prev_approved,
    COALESCE(p.prev_refused,0)   AS prev_refused,
    p.prev_approved::float / NULLIF(p.prev_app_count, 0)            AS prev_approval_rate,
    COALESCE(p.prev_apps_365d,0) AS prev_apps_365d, COALESCE(p.prev_apps_90d,0) AS prev_apps_90d,
    COALESCE(p.prev_refused_365d,0) AS prev_refused_365d,
    p.prev_avg_credit, p.days_since_last_application,
    -- payment behaviour
    COALESCE(i.inst_count,0) AS inst_count, i.inst_late_rate, i.inst_avg_days_late, i.inst_max_days_late,
    COALESCE(i.inst_missed_count,0) AS inst_missed_count, i.inst_payment_ratio,
    i.inst_late_rate_12m, i.inst_late_rate_prior,
    i.inst_late_rate_12m - i.inst_late_rate_prior                   AS inst_late_rate_change,
    COALESCE(i.inst_missed_12m,0) AS inst_missed_12m,
    -- POS / cash
    COALESCE(ps.pos_months,0) AS pos_months, COALESCE(ps.pos_dpd_months,0) AS pos_dpd_months,
    ps.pos_max_dpd, COALESCE(ps.pos_dpd_months_12m,0) AS pos_dpd_months_12m,
    -- credit card
    c.cc_util_avg, c.cc_util_max, c.cc_util_3m, c.cc_util_prior,
    c.cc_util_3m - c.cc_util_prior                                  AS cc_util_change,
    c.cc_balance_3m, c.cc_balance_prior,
    c.cc_balance_3m - c.cc_balance_prior                            AS cc_balance_change,
    COALESCE(c.cc_dpd_months,0) AS cc_dpd_months,
    -- exposure
    a.amt_credit                                                    AS current_exposure,
    a.amt_credit + COALESCE(b.bureau_debt_total,0) + COALESCE(c.cc_balance_3m,0) AS total_exposure,
    (a.amt_credit + COALESCE(b.bureau_debt_total,0)) / NULLIF(a.amt_income_total, 0) AS total_debt_to_income
FROM app a
LEFT JOIN bureau_agg b  USING (sk_id_curr)
LEFT JOIN bureau_bal bb USING (sk_id_curr)
LEFT JOIN prev_agg  p   USING (sk_id_curr)
LEFT JOIN inst_agg  i   USING (sk_id_curr)
LEFT JOIN pos_agg   ps  USING (sk_id_curr)
LEFT JOIN cc_agg    c   USING (sk_id_curr);

ALTER TABLE customer_360 ADD PRIMARY KEY (sk_id_curr);
CREATE UNIQUE INDEX ix_c360_customer ON customer_360 (customer_id);
CREATE INDEX ix_c360_vintage   ON customer_360 (vintage);
CREATE INDEX ix_c360_pop       ON customer_360 (population);
CREATE INDEX ix_c360_product   ON customer_360 (product);
CREATE INDEX ix_c360_segment   ON customer_360 (customer_segment);
ANALYZE customer_360;

-- =====================================================================
-- CUSTOMER x MONTH behaviour fact (months before application, -24..-1)
-- Pre-aggregated once so filtered behavioural trends are fast.
-- =====================================================================
DROP TABLE IF EXISTS customer_behaviour_monthly CASCADE;
CREATE TABLE customer_behaviour_monthly AS
WITH pos AS (
    SELECT sk_id_curr, months_balance AS month, MAX((sk_dpd > 0)::int) AS pos_dpd
    FROM raw_pos_cash WHERE months_balance BETWEEN -24 AND -1 GROUP BY 1, 2
), cc AS (
    SELECT sk_id_curr, months_balance AS month,
           SUM(amt_balance) AS cc_balance, SUM(amt_credit_limit_actual) AS cc_limit, MAX((sk_dpd > 0)::int) AS cc_dpd
    FROM raw_credit_card WHERE months_balance BETWEEN -24 AND -1 GROUP BY 1, 2
), inst AS (
    SELECT sk_id_curr, FLOOR(days_instalment / 30.44)::int AS month,
           COUNT(*) AS inst_n, SUM((days_entry_payment > days_instalment)::int) AS inst_late
    FROM raw_installments WHERE days_instalment BETWEEN -730 AND -1 GROUP BY 1, 2
)
SELECT COALESCE(p.sk_id_curr, c.sk_id_curr, i.sk_id_curr) AS sk_id_curr,
       COALESCE(p.month, c.month, i.month)                AS month,
       p.pos_dpd, c.cc_balance, c.cc_limit, c.cc_dpd, i.inst_n, i.inst_late
FROM pos p
FULL JOIN cc c   ON c.sk_id_curr = p.sk_id_curr AND c.month = p.month
FULL JOIN inst i ON i.sk_id_curr = COALESCE(p.sk_id_curr, c.sk_id_curr) AND i.month = COALESCE(p.month, c.month);
CREATE INDEX ix_cbm_customer ON customer_behaviour_monthly (sk_id_curr, month);
ANALYZE customer_behaviour_monthly;
