@echo off
call .venv\Scripts\activate
python src\train_obstacles_yolo.py --data data\datasets\obstacles\data.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --batch 4 --device cpu
pause
