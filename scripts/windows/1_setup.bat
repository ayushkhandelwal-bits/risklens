@echo off
REM RiskLens - one-time Python setup (run from the project root: scripts\windows\1_setup.bat)
cd /d "%~dp0\..\.."
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
if not exist .env copy .env.example .env
echo.
echo Setup complete. Next: scripts\windows\2_create_database.bat
goto :eof
:error
echo Setup failed - see the message above.
exit /b 1
