@echo off
REM RiskLens - create the PostgreSQL user, database and the read-only AI role.
REM Requires PostgreSQL 14+ installed with psql on PATH. You will be asked for the 'postgres' password.
cd /d "%~dp0\..\.."
psql -U postgres -c "CREATE USER risklens WITH PASSWORD 'risklens';"
psql -U postgres -c "CREATE DATABASE risklens OWNER risklens;"
psql -U postgres -d risklens -c "CREATE ROLE risklens_ai LOGIN PASSWORD 'risklens_ai';"
psql -U postgres -d risklens -c "ALTER ROLE risklens_ai SET default_transaction_read_only = on;"
psql -U postgres -d risklens -c "ALTER ROLE risklens_ai SET statement_timeout = '5s';"
echo.
echo Database ready. Next: scripts\windows\3_build.bat
