@echo off
REM RiskLens - run the automated test suite
cd /d "%~dp0\..\.."
call .venv\Scripts\activate.bat
python -m pytest tests -q
