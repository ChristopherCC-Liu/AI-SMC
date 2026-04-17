@echo off
REM AI-SMC Dashboard Web Server Launcher (Windows VPS)
REM 启动 localhost:8765 的 dashboard server
REM 然后在浏览器打开 http://localhost:8765

cd /d %~dp0\..

echo =========================================
echo  AI-SMC Dashboard Web Server
echo  URL: http://localhost:8765
echo  按 Ctrl+C 停止
echo =========================================
echo.

REM 激活 venv (如果存在 .venv)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python scripts\dashboard_server.py

pause
