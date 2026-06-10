@echo off
chcp 65001 >nul 2>&1
title F5 Gender & Race Labeling Tool
cd /d "%~dp0"

echo ============================================
echo   F5 性别与人种自动标注工具
echo ============================================
echo.

REM 修改下面的路径为你的实际路径
set INPUT_DIR=D:\F5MTL\person_scene_seg\test
set OUTPUT_FILE=D:\F5MTL\person_scene_seg\test\gender_race_labels.json

echo 输入目录: %INPUT_DIR%
echo 输出文件: %OUTPUT_FILE%
echo.

python label_gender_race.py -i "%INPUT_DIR%" -o "%OUTPUT_FILE%" -d opencv -c 0.6

echo.
pause
