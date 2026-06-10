@echo off
chcp 65001 >nul 2>&1
title F5 Gender & Race Labeling Tool
cd /d "%~dp0"

echo ============================================
echo   F5 性别与人种自动标注工具 - 一键安装
echo ============================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 安装依赖
echo [1/2] 安装依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)
echo [OK] 依赖安装完成
echo.

REM 运行标注
echo ============================================
echo   开始标注
echo ============================================
echo.

REM ========== 在这里修改你的输入输出路径 ==========
set INPUT_DIR=D:\F5MTL\person_scene_seg\test
set OUTPUT_FILE=D:\F5MTL\person_scene_seg\test\labels.json
REM ================================================

echo 输入目录: %INPUT_DIR%
echo 输出文件: %OUTPUT_FILE%
echo.

python label_gender_race.py -i "%INPUT_DIR%" -o "%OUTPUT_FILE%" -d opencv -c 0.6

echo.
pause
