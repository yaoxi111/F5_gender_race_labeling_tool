@echo off
chcp 65001 >nul
echo ============================================
echo   F5 标注工具 v2 离线版 - 环境安装
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10-3.12
    pause
    exit /b 1
)

:: 安装依赖（从 PyPI）
echo [1/2] 安装 Python 依赖...
pip install -r "%~dp0requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple/
if errorlevel 1 (
    echo [WARN] 清华源失败，尝试默认源...
    pip install -r "%~dp0requirements.txt"
)

echo.
echo [2/2] 验证模型文件...
set MODELS=%~dp0models
if exist "%MODELS%\deepface\gender_model_weights.h5" (
    echo   [OK] gender_model_weights.h5
) else (
    echo   [MISSING] gender_model_weights.h5
)
if exist "%MODELS%\deepface\race_model_single_batch.h5" (
    echo   [OK] race_model_single_batch.h5
) else (
    echo   [MISSING] race_model_single_batch.h5
)
if exist "%MODELS%\deepface\retinaface.h5" (
    echo   [OK] retinaface.h5
) else (
    echo   [MISSING] retinaface.h5
)
if exist "%MODELS%\yolov8n.pt" (
    echo   [OK] yolov8n.pt
) else (
    echo   [MISSING] yolov8n.pt
)

echo.
echo ============================================
echo   安装完成！运行示例：
echo   python "%~dp0scripts\label_gender_race.py" -i <图片文件夹> -o <输出.json> -d retinaface
echo ============================================
pause
