@echo off
REM RiskLens - start the API (port 8000) and the dashboard (port 8501) in two windows.
cd /d "%~dp0\..\.."
start "RiskLens API" cmd /k ".venv\Scripts\activate.bat && uvicorn backend.main:app --port 8000"
timeout /t 5 /nobreak >nul
start "RiskLens Dashboard" cmd /k ".venv\Scripts\activate.bat && streamlit run streamlit_app.py --server.port 8501"
echo Dashboard: http://localhost:8501     API docs: http://localhost:8000/docs
