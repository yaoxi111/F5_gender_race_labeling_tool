# F5 性别与人种自动标注工具 v2 — 技术文档

## 1. 项目概述

### 1.1 目标

为 F5 人物场景分割项目提供自动化的 **人体 + 人脸属性标注** 能力：
- **人体检测**：定位图片中所有人体的位置（YOLOv8）
- **人脸检测**：定位图片中所有人脸的位置（RetinaFace / MTCNN / OpenCV）
- **性别分类**：判定人脸的性别（Male / Female）
- **人种分类**：判定人脸的种族（Asian / White / Black / Indian / Middle Eastern / Latino）

输出与 F5 ODOT `FaceInfo` 结构对齐的 JSON 标注文件，可直接用于下游训练管线。

### 1.2 技术选型

| 组件 | 技术方案 | 说明 |
|------|----------|------|
| 人体检测 | YOLOv8 Nano (ultralytics) | COCO 预训练，class 0 = person |
| 人脸检测 | DeepFace 内置检测器 | RetinaFace / MTCNN / OpenCV 可选 |
| 性别分类 | DeepFace 预训练 VGG 模型 | 基于 VGG-Face 微调，准确率 ~97% |
| 人种分类 | DeepFace 预训练 VGG 模型 | 6 分类 |
| 图像处理 | OpenCV (cv2) | 图片读取、绘图、编码 |
| 数据处理 | NumPy | 数组运算、图片解码 |

---

## 2. 系统架构

### 2.1 整体流程

```
输入图片文件夹
      │
      ▼
┌─────────────┐
│  图片扫描    │  递归遍历目录，过滤图片格式
│ collect_images│
└──────┬──────┘
       │
       ▼ (逐张处理，独立 try-catch)
┌──────────────────────────────────────────────────┐
│                                                  │
│  ┌─────────────────┐                             │
│  │ YOLOv8 人体检测  │→ person bounding box       │
│  └─────────────────┘                             │
│                                                  │
│  ┌─────────────────────────────────────────┐     │
│  │       DeepFace.analyze()                │     │
│  │  ┌──────────┐ ┌──────────┐ ┌────────┐  │     │
│  │  │ 人脸检测  │→│ 人脸对齐  │→│ 属性分类│  │     │
│  │  │(Detector) │ │ (Align)  │ │(Gender │  │     │
│  │  └──────────┘ └──────────┘ │ +Race) │  │     │
│  │                            └────────┘  │     │
│  └─────────────────────────────────────────┘     │
│                                                  │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   后处理 & 过滤      │
            │  - 置信度过滤        │
            │  - 小人脸过滤(<30px) │
            │  - 枚举映射          │
            └────────┬────────────┘
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
    ┌──────────────┐   ┌──────────────┐
    │  JSON 标注    │   │  可视化图片   │
    │  输出文件     │   │  (可选)      │
    └──────────────┘   └──────────────┘
```

### 2.2 核心模块

| 模块 | 函数 | 职责 |
|------|------|------|
| 图片采集 | `collect_images()` | 递归扫描目录，收集图片路径 |
| 人体检测 | `detect_persons()` | YOLOv8 检测人体区域 |
| 人脸分析 | `analyze_image()` | DeepFace 检测人脸 + 性别/人种分类 |
| 可视化 | `draw_results()` | 绘制人体框（绿色）+ 人脸框（蓝/红/紫） |
| 主流程 | `run_labeling()` | 编排标注流程，含崩溃保护和检查点 |
| 入口 | `main()` | 命令行参数解析 |

---

## 3. 技术栈详解

### 3.1 YOLOv8 — 人体检测

[YOLOv8](https://github.com/ultralytics/ultralytics) 是 Ultralytics 推出的目标检测模型，本工具使用 Nano 版本（yolov8n.pt，6MB）。

**核心特点**：
- 基于 COCO 数据集预训练，包含 80 个目标类别
- 本工具只使用 class 0（person）进行人体检测
- 速度快（~1-2 fps），精度高
- 内置 NMS（非极大值抑制）去重

**调用方式**：
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model(img, conf=0.5, verbose=False, classes=[0])  # 只检测 person
```

**返回结构**：
```python
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # 人体框坐标
        score = box.conf[0]            # 置信度
```

### 3.2 DeepFace — 人脸属性分析

[DeepFace](https://github.com/serengil/deepface) 是一个轻量级的人脸识别与属性分析 Python 库。

**核心 API**：
```python
DeepFace.analyze(
    img_path=img,
    actions=["gender", "race"],
    detector_backend="retinaface",
    enforce_detection=False,
    silent=True,
)
```

**返回结构**：
```python
{
    "region": {"x": 100, "y": 50, "w": 200, "h": 250},  # 人脸框
    "dominant_gender": "Male",
    "gender": {"Man": 95.2, "Woman": 4.8},
    "dominant_race": "White",
    "race": {"white": 85.0, "asian": 10.0, ...}
}
```

### 3.3 人脸检测后端对比

| 后端 | 算法 | 速度 | 精度 | 模型大小 |
|------|------|------|------|----------|
| `opencv` | Haar Cascade | ★★★★★ | ★★★ | 内置 |
| `mtcnn` | MTCNN | ★★★ | ★★★★ | ~2MB |
| `retinaface` | RetinaFace | ★★ | ★★★★★ | ~119MB |

**RetinaFace**：基于 FPN 的多尺度人脸检测，同时预测人脸框和 5 个面部关键点，在 WIDER FACE 数据集上达到 SOTA 精度。

### 3.4 分类模型

| 模型 | 架构 | 输入 | 输出 | 准确率 |
|------|------|------|------|--------|
| 性别分类 | VGG-Face 微调 | 224×224 RGB | Woman/Man 概率 | ~97.44% |
| 人种分类 | VGG-Face 微调 | 224×224 RGB | 6 类概率分布 | ~90% |

### 3.5 OpenCV (cv2)

- 图片读取：`cv2.imdecode()` + `np.fromfile()` 支持中文路径
- 图片编码：`cv2.imencode()` 保存可视化结果
- 绘图：`cv2.rectangle()` / `cv2.putText()` 绘制框和标签

---

## 4. 数据处理流程

### 4.1 图片读取

```python
# Windows 中文路径兼容
raw = np.fromfile(img_path, dtype=np.uint8)
img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
```

### 4.2 人体检测

```python
# YOLOv8 检测人体
persons = detect_persons(img)  # 返回 [(x1, y1, x2, y2, score), ...]
```

### 4.3 人脸检测 + 分类

```python
# DeepFace.analyze() 内部流程：
# 1. 检测人脸区域 (detector_backend)
# 2. 人脸对齐（仿射变换，矫正旋转）
# 3. 裁剪 + 归一化到 224×224
# 4. 性别分类 (VGG 模型)
# 5. 人种分类 (VGG 模型)
```

### 4.4 过滤逻辑

```python
# 小人脸过滤
if w < 30 or h < 30:
    continue

# 置信度过滤
if gender_conf < conf_threshold:
    gender_id = 2  # Unknown
if race_conf < conf_threshold:
    race_id = 6  # Unknown
```

---

## 5. 崩溃保护机制（v2 新增）

针对大规模数据集（几万~几十万张）设计：

| 机制 | 实现方式 | 说明 |
|------|----------|------|
| 单图异常隔离 | 每张图独立 try-catch | 一张崩了不影响后续 |
| 定期检查点 | 每 N 张保存中间 JSON | 默认 50 张，可配置 |
| Ctrl+C 中断保护 | KeyboardInterrupt 捕获 | 中断时保存已处理结果 |
| 内存释放 | del + gc.collect() | 每张图处理后释放 |
| 错误统计 | errors 字段 | 输出 JSON 记录失败数 |

---

## 6. 输出格式

### 6.1 JSON 结构

```json
{
  "metadata": {
    "tool": "F5 Gender & Race Auto-Labeling Tool v2",
    "person_detector": "yolov8n",
    "face_detector_backend": "retinaface",
    "conf_threshold": 0.6,
    "total_images": 18,
    "total_persons": 18,
    "total_faces": 17,
    "errors": 0,
    "elapsed_seconds": 125.3,
    "fps": 0.14
  },
  "statistics": { ... },
  "labels": [
    {
      "image_path": "photo001.jpg",
      "person_count": 1,
      "persons": [{"x1": 50, "y1": 30, "x2": 400, "y2": 600, "label": "person", "det_score": 0.92}],
      "face_count": 1,
      "faces": [{"x1": 101, "y1": 141, "x2": 539, "y2": 579, "gender_label": "Female", "race_label": "White", ...}]
    }
  ]
}
```

### 6.2 枚举映射

**性别 (gender)**：0=Female, 1=Male, 2=Unknown

**人种 (race)**：0=Asian, 1=White, 2=Middle Eastern, 3=Indian, 4=Latino, 5=Black, 6=Unknown

---

## 7. 可视化模块

### 7.1 双层绘制

| 层级 | 颜色 | 内容 |
|------|------|------|
| 底层 | 绿色 (0,255,0) | 人体框 + "person" 标签 |
| 上层 | 蓝色 (255,128,0) | 男性人脸框 + 性别/人种标签 |
| 上层 | 红色 (0,80,255) | 女性人脸框 + 性别/人种标签 |
| 上层 | 紫色 (180,0,180) | 未知性别人脸框 |

### 7.2 标签格式

- 人体：`person`
- 人脸第一行：`性别/人种`（如 `Male/White`）
- 人脸第二行：`性别置信度%/人种置信度%`（如 `98%/99%`）

---

## 8. 模型文件

| 文件 | 大小 | 用途 | 位置 |
|------|------|------|------|
| `yolov8n.pt` | 6MB | YOLOv8 人体检测 | 工具根目录 |
| `gender_model_weights.h5` | 512MB | 性别分类 | `.deepface/weights/` |
| `race_model_single_batch.h5` | 512MB | 人种分类 | `.deepface/weights/` |
| `retinaface.h5` | 113MB | RetinaFace 人脸检测 | `.deepface/weights/` |

**总权重大小：约 1.14 GB**

通过环境变量 `DEEPFACE_HOME` 指向本地权重目录：
```python
os.environ["DEEPFACE_HOME"] = _SCRIPT_DIR
```

---

## 9. 离线部署方案

### 9.1 包结构

```
offline_package/
├── scripts/label_gender_race.py   # 主脚本（离线增强版）
├── packages/                      # 80+ 个 pip 离线包 (~1GB)
├── models/                        # 4 个 AI 模型 (~1.14GB)
├── install.bat / install.sh       # 一键安装脚本
└── README.md                      # 部署教程
```

**总大小：约 1.8 GB**

### 9.2 部署流程

```
1. 拷贝 offline_package 到目标服务器
2. pip install --no-index --find-links=packages -r requirements.txt
3. 运行标注工具
```

无需联网，所有依赖和模型已内置。

---

## 10. 性能指标

### 10.1 检测速度

| 组件 | 速度 | 说明 |
|------|------|------|
| YOLOv8 人体检测 | ~1-2 fps | CPU 模式 |
| DeepFace 人脸检测 (opencv) | ~1-3 fps | 最快 |
| DeepFace 人脸检测 (retinaface) | ~0.1-0.3 fps | 精度最高 |

### 10.2 分类精度

| 任务 | 准确率 | 说明 |
|------|--------|------|
| 性别分类 | ~97.44% | 二分类 |
| 人种分类 | ~90% | 6 分类，部分类别易混淆 |

---

## 11. 已知限制

1. **人种分类准确率有限**：6 类人种分类本身难度较高
2. **非正脸影响**：侧脸、低头等角度影响精度
3. **遮挡问题**：口罩、墨镜降低准确率
4. **动漫/CG 图片**：模型基于真人训练，不适用
5. **小人脸**：<30px 的人脸被过滤
6. **密集人群**：遮挡人脸可能漏检

---

## 12. 依赖清单

```
deepface>=0.0.93        # 人脸分析框架
opencv-python>=4.8.0    # 图像处理
tf-keras                # TensorFlow Keras
numpy                   # 数组运算
ultralytics             # YOLOv8 人体检测
torch>=1.8.0            # PyTorch（ultralytics 依赖）
```
