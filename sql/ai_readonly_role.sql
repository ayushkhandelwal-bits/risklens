-- =====================================================================
-- Least-privilege database role for the AI Risk Analyst's SQL tool.
-- The agent can only SELECT from analytical objects; it cannot read the
-- audit log or write anything. (Defence in depth on top of the SQL
-- validator and the READ ONLY transaction in ai/tools/sql_tool.py.)
-- Run as a superuser / database owner:  psql -f sql/ai_readonly_role.sql
-- =====================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'risklens_ai') THEN
        CREATE ROLE risklens_ai LOGIN PASSWORD 'risklens_ai';
    END IF;
END $$;

\i sql/ai_readonly_grants.sql
ALTER ROLE risklens_ai SET default_transaction_read_only = on;
ALTER ROLE risklens_ai SET statement_timeout = '5s';
