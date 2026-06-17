# F5 标注工具 v2 — 离线部署包

## 目录结构

```
offline_package/
├── install.bat                    # Windows 一键安装
├── install.sh                     # Linux 一键安装
├── requirements.txt               # Python 依赖清单
├── README.md                      # 本文档
├── scripts/
│   └── label_gender_race.py       # 主脚本（离线增强版）
├── packages/                      # pip 离线包（约 1GB）
│   ├── tensorflow-*.whl
│   ├── torch-*.whl
│   └── ... (80+ 个 .whl 文件)
└── models/
    ├── yolov8n.pt                 # YOLOv8 人体检测模型 (6MB)
    └── deepface/
        ├── gender_model_weights.h5    # 性别分类模型 (512MB)
        ├── race_model_single_batch.h5 # 人种分类模型 (512MB)
        └── retinaface.h5              # RetinaFace 人脸检测 (113MB)
```

**总大小：约 1.8 GB**

## 部署步骤

### 1. 拷贝整个 `offline_package` 文件夹到目标服务器

### 2. 安装 Python 依赖（完全离线）

```bash
pip install --no-index --find-links=packages -r requirements.txt
```

或双击 `install.bat`（Windows）/ `bash install.sh`（Linux）。

> ✅ 所有依赖已内置在 `packages/` 目录中，无需联网。

### 3. 运行标注

```bash
python scripts/label_gender_race.py -i <图片文件夹> -o <输出.json> -d retinaface -v <可视化目录>
```

## 参数说明

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入图片文件夹 | 必填 |
| `--output` | `-o` | 输出 JSON 路径 | `./output/labels.json` |
| `--conf` | `-c` | 置信度阈值 | `0.6` |
| `--detector` | `-d` | 人脸检测后端 | `opencv` |
| `--viz-dir` | `-v` | 可视化输出目录 | — |
| `--checkpoint` | — | 检查点间隔（每 N 张保存） | `50` |

## 使用示例

```bash
# 推荐：retinaface + 可视化
python scripts/label_gender_race.py -i /data/images -o /data/labels.json -d retinaface -v /data/viz

# 大规模数据集：降低检查点频率
python scripts/label_gender_race.py -i /data/images -o /data/labels.json -d retinaface --checkpoint 100

# 快速模式：opencv 最快
python scripts/label_gender_race.py -i /data/images -o /data/labels.json -d opencv
```

## 大规模数据集保护机制

离线版包含以下保护：

1. **单图异常隔离**：每张图片独立 try-catch，一张失败不影响后续
2. **定期检查点**：每 N 张自动保存中间结果（默认 50 张）
3. **中断保护**：Ctrl+C 中断时自动保存已处理结果
4. **内存释放**：每张图片处理后释放内存，防止 OOM
5. **错误统计**：统计失败图片数量，记录在输出 JSON 中

## 模型说明

| 模型 | 大小 | 用途 |
|------|------|------|
| `yolov8n.pt` | 6MB | 人体检测（YOLOv8 Nano，COCO 预训练） |
| `gender_model_weights.h5` | 512MB | 性别分类（VGG-Face 微调） |
| `race_model_single_batch.h5` | 512MB | 人种分类（VGG-Face 微调，6 类） |
| `retinaface.h5` | 113MB | 人脸检测（RetinaFace，精度最高） |

所有模型已内置在 `models/` 目录中，无需联网下载。
