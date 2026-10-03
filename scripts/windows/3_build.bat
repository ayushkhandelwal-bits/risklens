@echo off
REM RiskLens - ETL + Customer 360 + models + SHAP + early warning + anomaly + monitoring (~3-6 min)
cd /d "%~dp0\..\.."
call .venv\Scripts\activate.bat
python -m scripts.build_all || (echo Build failed - see the message above. & exit /b 1)
echo.
echo Build complete. Next: scripts\windows\4_run.bat
