@echo off
call .venv\Scripts\activate
python src\train_sidewalk_segformer.py --data data\datasets\sidewalk --epochs 20 --batch 2
pause
