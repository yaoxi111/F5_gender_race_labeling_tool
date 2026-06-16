#!/bin/bash
echo "============================================"
echo "  F5 标注工具 v2 离线版 - 环境安装"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] 未找到 Python3，请先安装 Python 3.10-3.12"
    exit 1
fi

# 安装依赖
echo "[1/2] 安装 Python 依赖..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple/ || \
pip3 install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "[2/2] 验证模型文件..."
MODELS="$SCRIPT_DIR/models"
for f in "deepface/gender_model_weights.h5" "deepface/race_model_single_batch.h5" "deepface/retinaface.h5" "yolov8n.pt"; do
    if [ -f "$MODELS/$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
    fi
done

echo
echo "============================================"
echo "  安装完成！运行示例："
echo "  python3 $SCRIPT_DIR/scripts/label_gender_race.py -i <图片文件夹> -o <输出.json> -d retinaface"
echo "============================================"
