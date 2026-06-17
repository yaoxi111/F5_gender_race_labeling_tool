# F5 标注工具 v2 — 离线部署包使用教程

本包为 **完全离线** 环境设计，包含工具运行所需的全部文件（Python 依赖包 + AI 模型权重），无需联网即可部署使用。

---

## 1. 包含内容

```
offline_package/
├── install.bat                    # Windows 一键安装脚本
├── install.sh                     # Linux 一键安装脚本
├── requirements.txt               # Python 依赖清单
├── README.md                      # 本文档
├── scripts/
│   └── label_gender_race.py       # 主程序（离线增强版，含崩溃保护）
├── packages/                      # pip 离线安装包（约 1GB）
│   ├── tensorflow-2.21.0-*.whl    #   TensorFlow 深度学习框架
│   ├── torch-2.12.0-*.whl         #   PyTorch 深度学习框架
│   ├── ultralytics-8.4.68-*.whl   #   YOLOv8 人体检测
│   ├── deepface-0.0.100-*.whl     #   DeepFace 人脸分析
│   ├── opencv_python-*.whl        #   OpenCV 图像处理
│   └── ... (共 80+ 个依赖包)
└── models/
    ├── yolov8n.pt                 # YOLOv8 人体检测模型 (6MB)
    └── deepface/
        ├── gender_model_weights.h5    # 性别分类模型 (512MB)
        ├── race_model_single_batch.h5 # 人种分类模型 (512MB)
        └── retinaface.h5              # RetinaFace 人脸检测模型 (113MB)
```

**总大小：约 1.8 GB**

---

## 2. 环境要求

- **操作系统**：Windows 10/11 或 Linux（Ubuntu 18.04+）
- **Python**：3.10、3.11 或 3.12（推荐 3.12）
- **磁盘空间**：至少 5 GB（包本身 1.8GB + 安装后约 3GB）
- **内存**：建议 8GB 以上（处理大批量时）
- **网络**：**不需要**（完全离线安装）

---

## 3. 部署步骤

### 步骤 1：拷贝文件夹

将整个 `offline_package` 文件夹拷贝到目标服务器。

方式任选：
- U 盘拷贝
- 内网文件传输
- 移动硬盘
- scp/rsync 远程拷贝

```bash
# 示例：scp 远程拷贝
scp -r offline_package/ user@server:/home/user/

# 示例：rsync 远程拷贝
rsync -avz offline_package/ user@server:/home/user/offline_package/
```

### 步骤 2：安装 Python 依赖

**Windows（推荐双击 install.bat）：**
```cmd
cd offline_package
install.bat
```

**Linux：**
```bash
cd offline_package
chmod +x install.sh
./install.sh
```

**或手动安装：**
```bash
cd offline_package
pip install --no-index --find-links=packages -r requirements.txt
```

> `--no-index` 表示不从网络下载，`--find-links=packages` 指定从本地 `packages/` 目录查找安装包。

### 步骤 3：验证安装

```bash
python -c "import deepface; import cv2; import ultralytics; import torch; print('All OK')"
```

输出 `All OK` 表示环境就绪。

---

## 4. 使用方法

### 4.1 基本用法

```bash
python scripts/label_gender_race.py -i <图片文件夹> -o <输出JSON> -d retinaface -v <可视化目录>
```

**参数说明：**

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入图片文件夹路径 | **必填** |
| `--output` | `-o` | 输出 JSON 文件路径 | `./output/labels.json` |
| `--conf` | `-c` | 置信度阈值（0~1，低于此值标为 Unknown） | `0.6` |
| `--detector` | `-d` | 人脸检测后端 | `opencv` |
| `--viz-dir` | `-v` | 可视化输出目录（保存带框标注图片） | — |
| `--checkpoint` | — | 检查点间隔（每 N 张自动保存） | `50` |

### 4.2 使用示例

**示例 1：推荐用法（精度最高）**
```bash
python scripts/label_gender_race.py -i /data/images -o /data/labels.json -d retinaface -v /data/viz
```

**示例 2：大批量快速标注**
```bash
python scripts/label_gender_race.py -i /data/images -o /data/labels.json -d opencv
```

**示例 3：大规模数据集（自定义检查点）**
```bash
python scripts/label_gender_race.py -i /data/images -o /data/labels.json -d retinaface --checkpoint 100
```

**示例 4：高精度精标（严格阈值）**
```bash
python scripts/label_gender_race.py -i /data/images -o /data/labels.json -d retinaface -c 0.8
```

### 4.3 检测后端选择

| 后端 | 速度 | 精度 | 适用场景 |
|------|------|------|----------|
| `opencv` | ★★★★★ | ★★★ | 大批量预标注（几千张以上） |
| `mtcnn` | ★★★ | ★★★★ | 日常标注（精度/速度平衡） |
| `retinaface` | ★★ | ★★★★★ | 精标/小批量（推荐） |

### 4.4 输出说明

工具输出一个 JSON 文件，包含：

```json
{
  "metadata": {
    "tool": "F5 Gender & Race Auto-Labeling Tool v2 (offline)",
    "person_detector": "yolov8n",
    "face_detector_backend": "retinaface",
    "total_images": 1000,
    "total_persons": 1200,
    "total_faces": 1150,
    "errors": 2
  },
  "labels": [
    {
      "image_path": "photo001.jpg",
      "person_count": 1,
      "persons": [{"x1": 50, "y1": 30, "x2": 400, "y2": 600, "label": "person"}],
      "face_count": 1,
      "faces": [{"x1": 100, "y1": 50, "x2": 300, "y2": 250, "gender_label": "Male", "race_label": "White"}]
    }
  ]
}
```

- **persons**：人体检测结果（绿色框，类别为 `person`）
- **faces**：人脸检测结果（含性别、人种分类）
- **errors**：处理失败的图片数量

---

## 5. 大规模数据集保护

离线增强版包含以下保护机制，适合处理几万甚至几十万张图片：

| 保护机制 | 说明 |
|----------|------|
| 单图异常隔离 | 每张图片独立 try-catch，一张崩了不影响后续 |
| 定期检查点 | 每 N 张自动保存中间结果到输出 JSON（默认 50 张） |
| Ctrl+C 中断保护 | 中断时自动保存已处理结果，不丢失 |
| 内存释放 | 每张图处理后释放内存，防止 OOM |
| 错误统计 | 输出 JSON 中记录失败图片数 |

**如果中途崩溃**：输出文件中会包含已处理的图片结果，重新运行即可继续（会覆盖之前的输出）。

---

## 6. 检测后端性能参考

| 后端 | 人脸检测速度 | 人脸框精度 | 说明 |
|------|-------------|-----------|------|
| opencv | ~1-3 fps | 一般 | 最快，适合大批量 |
| mtcnn | ~0.5-1 fps | 良好 | 精度/速度平衡 |
| retinaface | ~0.1-0.3 fps | 最优 | 推荐，精度最高 |

> 人体检测（YOLOv8）速度约 1-2 fps，与人脸检测后端独立。

---

## 7. 常见问题

### Q: 安装时报错 `No matching distribution found`？

确认使用了 `--no-index --find-links=packages` 参数，且 `packages/` 目录存在。

### Q: Python 版本不匹配？

离线包中的 `.whl` 文件是针对 Python 3.12 编译的。如果目标服务器是其他版本，需要在有网络的同版本环境中重新下载：
```bash
pip download -r requirements.txt -d packages/ --only-binary=:all:
```

### Q: 内存不足（OOM）？

- 减小检查点间隔：`--checkpoint 20`
- 关闭可视化（不加 `-v` 参数）
- 使用 `opencv` 后端（内存占用最小）

### Q: 处理速度太慢？

- 使用 `opencv` 后端：`-d opencv`（最快）
- 关闭可视化（不加 `-v`）
- 确保服务器有足够的 CPU 核心

### Q: 如何查看处理进度？

输出 JSON 中的 `metadata.total_images` 会实时更新（检查点保存时），可以用 `tail -f` 或定期读取查看进度。

---

## 8. 可视化颜色说明

生成的可视化图片中：

| 颜色 | 含义 |
|------|------|
| 绿色框 | 人体区域（person） |
| 蓝色框 | 男性人脸 |
| 红色框 | 女性人脸 |
| 紫色框 | 未知性别人脸 |

标签格式：`性别/人种` + `置信度%`（如 `Male/White 98%/99%`）
