@echo off
echo Starting KINESYS...
"%~dp0.venv\Scripts\python.exe" -u "%~dp0main.py" %*
pause
