@echo off
REM AI-SMC Dashboard Web Server Launcher (Windows VPS).
REM Task Scheduler OnStart task = \AI-SMC-DashboardWeb.
REM 启动 localhost:8765 dashboard server，浏览器打开 http://localhost:8765

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\dashboard_server.py >> logs\dashboard_web_stdout.log 2>> logs\dashboard_web_stderr.log
