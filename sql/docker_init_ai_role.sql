-- Executed once by the postgres container on first start (superuser context).
CREATE ROLE risklens_ai LOGIN PASSWORD 'risklens_ai';
ALTER ROLE risklens_ai SET default_transaction_read_only = on;
ALTER ROLE risklens_ai SET statement_timeout = '5s';
