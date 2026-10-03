-- Re-applied after every build (dropping/recreating tables removes grants).
-- Runs as the application/owner role; the role itself is created once by sql/ai_readonly_role.sql.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'risklens_ai') THEN
    REVOKE ALL ON ALL TABLES IN SCHEMA public FROM risklens_ai;
    GRANT USAGE ON SCHEMA public TO risklens_ai;
    GRANT SELECT ON
        customer_360, customer_behaviour_monthly,
        v_portfolio, v_portfolio_by_vintage, v_investigation_queue, mv_behaviour_trend,
        risk_scores, customer_explanations, global_feature_importance,
        early_warning_signals, ews_triggers, anomaly_scores, anomaly_rules,
        model_registry, model_monitoring, data_quality_checks
    TO risklens_ai;
  END IF;
END $$;
