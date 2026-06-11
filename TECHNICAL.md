# F5 性别与人种自动标注工具 — 技术文档

## 1. 项目概述

### 1.1 目标

为 F5 人物场景分割项目提供自动化的人脸属性标注能力，对图片中的人脸进行：
- **人脸检测**：定位图片中所有人脸的位置（边界框）
- **性别分类**：判定人脸的性别（Male / Female）
- **人种分类**：判定人脸的种族（Asian / White / Black / Indian / Middle Eastern / Latino）

输出与 F5 ODOT `FaceInfo` 结构对齐的 JSON 标注文件，可直接用于下游训练管线。

### 1.2 技术选型

| 组件 | 技术方案 | 说明 |
|------|----------|------|
| 人脸检测 | DeepFace 内置检测器（RetinaFace / MTCNN / OpenCV） | 多后端可选，RetinaFace 精度最高 |
| 性别分类 | DeepFace 预训练 VGG 模型 | 基于 VGG-Face 微调，准确率 ~97% |
| 人种分类 | DeepFace 预训练 VGG 模型 | 6 分类（Asian/White/Black/Indian/Middle Eastern/Latino） |
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
       ▼ (逐张处理)
┌─────────────────────────────────────────┐
│           DeepFace.analyze()             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ 人脸检测  │→ │ 人脸对齐  │→ │ 属性分类│ │
│  │(Detector) │  │ (Align)  │  │(Gender │ │
│  │          │  │          │  │ +Race) │ │
│  └──────────┘  └──────────┘  └────────┘ │
└──────────────────┬──────────────────────┘
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
| 核心分析 | `analyze_image()` | 调用 DeepFace 完成检测+分类 |
| 可视化 | `draw_faces()` | 在图片上绘制人脸框和标签 |
| 主流程 | `run_labeling()` | 编排整个标注流程，统计结果 |
| 入口 | `main()` | 命令行参数解析 |

---

## 3. 技术栈详解

### 3.1 DeepFace 框架

[DeepFace](https://github.com/serengil/deepface) 是一个轻量级的人脸识别与属性分析 Python 库，封装了多种前沿人脸分析模型。

**本工具使用的核心 API**：

```python
DeepFace.analyze(
    img_path=img,                    # 输入图片（numpy array 或文件路径）
    actions=["gender", "race"],      # 分析任务
    detector_backend="retinaface",   # 检测后端
    enforce_detection=False,         # 未检测到人脸时不抛异常
    silent=True,                     # 静默模式
)
```

**返回结构**：
```python
{
    "region": {"x": 100, "y": 50, "w": 200, "h": 250},  # 人脸框
    "dominant_gender": "Male",          # 性别
    "gender": {"Man": 95.2, "Woman": 4.8},  # 性别概率
    "dominant_race": "White",           # 人种
    "race": {"white": 85.0, "asian": 10.0, ...}  # 人种概率
}
```

### 3.2 人脸检测后端

DeepFace 支持多种人脸检测后端，本工具支持以下 7 种：

| 后端 | 算法 | 速度 | 精度 | 模型大小 | 说明 |
|------|------|------|------|----------|------|
| `opencv` | Haar Cascade | ★★★★★ | ★★★ | 内置 | OpenCV 经典级联分类器，速度最快 |
| `ssd` | SSD + ResNet | ★★★★ | ★★★ | ~10MB | 单次多尺度检测 |
| `mediapipe` | BlazeFace | ★★★★ | ★★★★ | ~1MB | Google 轻量级人脸检测 |
| `mtcnn` | MTCNN | ★★★ | ★★★★ | ~2MB | 三阶级联网络（P-Net/R-Net/O-Net） |
| `fastmtcnn` | FastMTCNN | ★★★★ | ★★★★ | ~2MB | MTCNN 优化版本 |
| `yolov8n` | YOLOv8 Nano | ★★★★ | ★★★★ | ~6MB | Ultralytics 轻量检测 |
| `retinaface` | RetinaFace | ★★ | ★★★★★ | ~119MB | **推荐**，精度最高，特征点定位最准 |

**RetinaFace 简介**：
- 基于特征金字塔网络（FPN）的多尺度人脸检测
- 同时预测人脸框和 5 个面部关键点（双眼、鼻尖、嘴角）
- 在 WIDER FACE 数据集上达到 SOTA 精度
- 模型文件：`retinaface.h5`（119MB，首次使用时自动下载）

### 3.3 性别分类模型

- 架构：基于 VGG-Face 微调的二分类网络
- 输入：224×224 RGB 人脸图片（经对齐和归一化）
- 输出：`Woman` / `Man` 两个类别的概率
- 准确率：~97.44%
- 模型文件：`gender_model_weights.h5`（~512MB）

### 3.4 人种分类模型

- 架构：基于 VGG-Face 微调的 6 分类网络
- 输入：224×224 RGB 人脸图片
- 输出：6 个种族类别的概率分布
  - Asian（东亚/东南亚）
  - White（白人）
  - Black（黑人）
  - Indian（南亚/印度）
  - Middle Eastern（中东）
  - Latino Hispanic（拉丁裔）
- 模型文件：`race_model_single_batch.h5`（~512MB）

### 3.5 OpenCV (cv2)

用途：
- 图片读取：`cv2.imdecode()` + `np.fromfile()` 支持中文路径
- 图片编码：`cv2.imencode()` 保存可视化结果
- 绘图：`cv2.rectangle()` / `cv2.putText()` 绘制人脸框和标签

### 3.6 NumPy

用途：
- 图片解码：`np.fromfile()` 读取原始字节
- 数组类型处理：JSON 序列化时的类型转换

---

## 4. 数据处理流程

### 4.1 图片读取（支持中文路径）

```python
# Windows 中文路径兼容方案
raw = np.fromfile(img_path, dtype=np.uint8)  # 读取原始字节
img = cv2.imdecode(raw, cv2.IMREAD_COLOR)     # 解码为 BGR 图片
```

### 4.2 人脸检测 + 分类

```python
# DeepFace.analyze() 内部流程：
# 1. 检测人脸区域 (detector_backend)
# 2. 人脸对齐（仿射变换，矫正旋转）
# 3. 裁剪 + 归一化到 224×224
# 4. 性别分类 (VGG 模型)
# 5. 人种分类 (VGG 模型)
```

### 4.3 置信度过滤

```python
# 性别
if gender_conf < conf_threshold:
    gender_id = 2  # Unknown
else:
    gender_id = GENDER_MAP.get(gender_label, 2)

# 人种
if race_conf < conf_threshold:
    race_id = 6  # Unknown
else:
    race_id = RACE_MAP.get(race_label.lower(), 6)
```

### 4.4 小人脸过滤

```python
# 过滤掉宽或高 < 30 像素的微小人脸
if w < 30 or h < 30:
    continue
```

---

## 5. 输出格式

### 5.1 JSON 结构

```json
{
  "metadata": {
    "tool": "F5 Gender & Race Auto-Labeling Tool",
    "model": "deepface",
    "detector_backend": "retinaface",
    "conf_threshold": 0.6,
    "total_images": 34,
    "total_faces": 35,
    "elapsed_seconds": 144.6,
    "fps": 0.23
  },
  "statistics": {
    "male": 19, "female": 16, "unknown_gender": 0,
    "asian": 0, "white": 18, "middle_eastern": 0,
    "indian": 0, "latino": 0, "black": 2, "unknown_race": 15,
    "no_face_images": 0
  },
  "labels": [
    {
      "image_path": "celeb_28102.jpg",
      "image_abs_path": "D:\\...\\celeb_28102.jpg",
      "face_count": 1,
      "faces": [
        {
          "x1": 101, "y1": 141, "x2": 539, "y2": 579,
          "det_score": 0.9908,
          "gender": 0,
          "gender_label": "Female",
          "gender_conf": 0.9908,
          "raw_gender_scores": {"Woman": 0.9908, "Man": 0.0092},
          "race": 1,
          "race_label": "White",
          "race_conf": 0.9912,
          "raw_race_scores": {"asian": 0.0, "white": 0.9912, ...}
        }
      ]
    }
  ]
}
```

### 5.2 枚举映射

**性别 (gender)**

| 值 | 标签 | 含义 |
|----|------|------|
| 0 | Female | 女性 |
| 1 | Male | 男性 |
| 2 | Unknown | 未知（置信度低于阈值） |

**人种 (race)**

| 值 | 标签 | 含义 |
|----|------|------|
| 0 | Asian | 东亚/东南亚人 |
| 1 | White | 白人 |
| 2 | Middle Eastern | 中东人 |
| 3 | Indian | 南亚/印度人 |
| 4 | Latino | 拉丁裔 |
| 5 | Black | 黑人 |
| 6 | Unknown | 未知（置信度低于阈值） |

---

## 6. 可视化模块

### 6.1 颜色方案

| 性别 | 框颜色 (BGR) | 说明 |
|------|-------------|------|
| Female | (0, 80, 255) 红色 | 女性 |
| Male | (255, 128, 0) 蓝色 | 男性 |
| Unknown | (180, 0, 180) 紫色 | 未知性别 |

### 6.2 标签格式

每张人脸框上方显示两行标签：
- 第一行：`性别/人种`（如 `Male/White`）
- 第二行：`性别置信度%/人种置信度%`（如 `98%/99%`）

### 6.3 自适应布局

- 标签默认显示在人脸框上方
- 若超出图片顶部边界，自动调整到框内显示

---

## 7. 模型文件说明

所有模型权重存放在 `.deepface/weights/` 目录：

| 文件 | 大小 | 用途 | 首次获取方式 |
|------|------|------|-------------|
| `gender_model_weights.h5` | ~512MB | 性别分类 | 已内置 |
| `race_model_single_batch.h5` | ~512MB | 人种分类 | 已内置 |
| `retinaface.h5` | ~119MB | RetinaFace 人脸检测 | 首次使用 `-d retinaface` 时自动下载 |

通过设置环境变量 `DEEPFACE_HOME` 指向工具目录，确保 DeepFace 使用本地权重：
```python
os.environ["DEEPFACE_HOME"] = _SCRIPT_DIR
```

---

## 8. 性能指标

### 8.1 测试环境

- CPU: Intel/AMD x86_64
- Python: 3.12
- TensorFlow: 2.x (tf-keras)

### 8.2 各后端性能对比

| 检测后端 | 检测速度 | 人脸框精度 | 首次加载 | 推荐场景 |
|----------|---------|-----------|---------|---------|
| opencv | ~1-3 fps | 一般 | 即时 | 大批量预标注 |
| mtcnn | ~0.5-1 fps | 良好 | ~2s | 日常标注 |
| retinaface | ~0.1-0.3 fps | 最优 | ~5s | 精标/小批量 |

### 8.3 分类模型精度

| 任务 | 准确率 | 说明 |
|------|--------|------|
| 性别分类 | ~97.44% | 二分类，整体准确率高 |
| 人种分类 | ~90% | 6 分类，部分类别容易混淆 |

---

## 9. 已知限制

1. **人种分类准确率有限**：6 类人种分类本身难度较高，部分类别（如 White vs Middle Eastern）容易混淆
2. **非正脸影响**：侧脸、低头、抬头等角度会影响检测和分类精度
3. **遮挡问题**：口罩、墨镜等遮挡会降低分类准确率
4. **动漫/CG 图片**：模型基于真人训练，对动漫、CG 角色不适用
5. **小人脸**：图片中过小的人脸（<30px）会被过滤，无法分类
6. **多人脸场景**：密集人群中的遮挡人脸可能漏检

---

## 10. 依赖清单

```
deepface>=0.0.93        # 人脸分析框架
opencv-python>=4.8.0    # 图像处理
tf-keras                # TensorFlow Keras（DeepFace 后端）
numpy                   # 数组运算
```

可选依赖（用于特定检测后端）：
```
mediapipe               # mediapipe 后端
ultralytics             # yolov8n 后端
mtcnn                   # mtcnn 后端（DeepFace 自动安装）
```
