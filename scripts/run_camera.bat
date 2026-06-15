@echo off
call .venv\Scripts\activate
python src\realtime_app.py --source camera --width 640 --height 360 --seg-every 3
pause
