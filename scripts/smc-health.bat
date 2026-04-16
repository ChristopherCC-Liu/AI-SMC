@echo off
set PYTHONUTF8=1
cd /d C:\AI-SMC
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');import MetaTrader5 as mt5;mt5.initialize();i=mt5.account_info();t=mt5.symbol_info_tick('XAUUSD');print(f'Account: {i.login} @ {i.server}');print(f'Balance: ${i.balance}');print(f'XAUUSD: {t.bid}/{t.ask}');mt5.shutdown()"
