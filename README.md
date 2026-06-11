# F5 性别与人种自动标注工具

基于 [deepface](https://github.com/serengil/deepface) 的人脸检测 + 性别分类 + 人种分类自动标注工具，输出与 F5 ODOT `FaceInfo` 结构对齐的 JSON 标注文件。

---

## 目录结构

```
F5_gender_race_labeling_tool/
├── label_gender_race.py        # 主脚本（唯一入口）
├── requirements.txt            # Python 依赖
├── setup_and_run.bat           # 一键安装依赖 + 运行
├── run_example.bat             # 示例运行脚本
├── README.md                   # 本文档
└── .deepface/weights/          # 模型权重（已内置，无需下载）
    ├── gender_model_weights.h5
    └── race_model_single_batch.h5
```

模型权重已内置在 `.deepface/weights/` 目录中，首次运行无需联网下载。

---

## 1. 环境安装

### 1.1 Python 要求

- Python 3.10 ~ 3.12（推荐 3.12）

### 1.2 安装依赖

```bash
cd C:\Users\李宗泽\Desktop\F5_gender_race_labeling_tool
pip install -r requirements.txt
```

### 1.3 模型权重

模型权重已内置在 `.deepface/weights/` 目录中，无需联网下载。首次运行直接可用。

---

## 2. 使用方法

### 2.1 命令格式

```bash
python label_gender_race.py --input <图片文件夹> --output <输出JSON> [选项]
```

### 2.2 参数说明

| 参数 | 缩写 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|------|--------|
| `--input` | `-i` | string | **是** | 输入图片文件夹路径 | — |
| `--output` | `-o` | string | 否 | 输出 JSON 文件路径 | `./output/gender_race_labels.json` |
| `--conf` | `-c` | float | 否 | 置信度阈值（低于此值标记为 Unknown） | `0.6` |
| `--detector` | `-d` | string | 否 | 人脸检测后端 | `opencv` |
| `--viz-dir` | `-v` | string | 否 | 可视化输出目录（保存带人脸框的图片） | — |

### 2.3 支持的图片格式

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`

工具会递归扫描输入文件夹下的所有子目录。

---

## 3. 使用示例

### 示例 1：基本用法

```bash
python label_gender_race.py -i D:\F5MTL\person_scene_seg\test -o D:\F5MTL\person_scene_seg\test\gender_race_labels.json
```

### 示例 2：高精度模式（适合小批量精标）

```bash
python label_gender_race.py -i D:\F5MTL\person_scene_seg\test -o D:\F5MTL\person_scene_seg\test\labels.json -d retinaface -c 0.8
```

### 示例 3：大批量快速标注

```bash
python label_gender_race.py -i D:\F5MTL\dataset\images -o D:\F5MTL\dataset\labels.json -d opencv -c 0.5
```

### 示例 4：输出到输入目录同级

```bash
python label_gender_race.py -i D:\data\my_photos
# 输出默认为 ./output/gender_race_labels.json
```

### 示例 5：生成可视化标注图片

在标注的同时，将人脸框和标签绘制到图片上并保存：

```bash
python label_gender_race.py -i D:\F5MTL\person_scene_seg\test2 -o D:\F5MTL\person_scene_seg\test2\labels.json -v D:\F5MTL\person_scene_seg\test2\viz
```

可视化图片中：
- **蓝色框** = 男性
- **红色框** = 女性
- **紫色框** = 未知性别
- 标签格式：`性别/人种` + `置信度%`

---

## 4. 检测后端选择指南

| 后端 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| `opencv` | ★★★★★ | ★★★ | 大批量预标注（几千张以上） |
| `mtcnn` | ★★★ | ★★★★ | 日常标注（精度/速度平衡） |
| `retinaface` | ★★ | ★★★★★ | 精标/小批量（精度最高） |
| `mediapipe` | ★★★★ | ★★★★ | 实时场景 |
| `ssd` | ★★★★ | ★★★ | 快速检测 |
| `yolov8n` | ★★★★ | ★★★★ | 均衡选择 |
| `fastmtcnn` | ★★★★ | ★★★★ | mtcnn 的加速版本 |

**推荐策略**：
1. 先用 `opencv` 快速跑一遍全部数据
2. 对 Unknown 样本用 `retinaface` 精标
3. 人工抽查 20~50 张验证准确率

---

## 5. 置信度阈值说明

置信度表示模型对分类结果的确定程度（0~1），性别和人种共用同一阈值。

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| `0.5` | 宽松，大部分直接标注 | 数据量少，希望尽量多标注 |
| `0.6` | **默认推荐**，平衡精度和召回 | 通用场景 |
| `0.8` | 严格，只保留高置信度 | 对标注质量要求高 |
| `0.9` | 非常严格 | 只要几乎确定的结果 |

---

## 6. 输出格式详解

### 6.1 JSON 整体结构

```json
{
  "metadata": { ... },
  "statistics": { ... },
  "labels": [ ... ]
}
```

### 6.2 metadata

```json
"metadata": {
  "tool": "F5 Gender & Race Auto-Labeling Tool",
  "model": "deepface",
  "detector_backend": "opencv",
  "conf_threshold": 0.6,
  "total_images": 5,
  "total_faces": 5,
  "elapsed_seconds": 227.85,
  "fps": 0.02
}
```

### 6.3 statistics

```json
"statistics": {
  "male": 3,
  "female": 1,
  "unknown_gender": 1,
  "asian": 0,
  "white": 3,
  "middle_eastern": 0,
  "indian": 0,
  "latino": 0,
  "black": 0,
  "unknown_race": 2,
  "no_face_images": 0
}
```

### 6.4 labels（逐图标注）

```json
"labels": [
  {
    "image_path": "celeb_28271.jpg",
    "image_abs_path": "D:\\F5MTL\\...\\celeb_28271.jpg",
    "face_count": 1,
    "faces": [
      {
        "x1": 0, "y1": 0, "x2": 639, "y2": 639,
        "det_score": 0.9915,
        "gender": 1,
        "gender_label": "Male",
        "gender_conf": 0.9915,
        "raw_gender_scores": {
          "Woman": 0.0085,
          "Man": 0.9915
        },
        "race": 1,
        "race_label": "White",
        "race_conf": 0.9984,
        "raw_race_scores": {
          "asian": 0.0,
          "indian": 0.0,
          "black": 0.0,
          "white": 0.9984,
          "middle eastern": 0.0003,
          "latino hispanic": 0.0014
        }
      }
    ]
  }
]
```

### 6.5 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `x1, y1, x2, y2` | int | 人脸框坐标（左上角、右下角） |
| `det_score` | float | 检测置信度 |
| `gender` | int | 性别枚举：0=Female, 1=Male, 2=Unknown |
| `gender_label` | string | 性别文本标签 |
| `gender_conf` | float | 性别置信度 (0~1) |
| `raw_gender_scores` | dict | 性别原始分数（Woman/Man 各自的概率） |
| `race` | int | 人种枚举（见下表） |
| `race_label` | string | 人种文本标签 |
| `race_conf` | float | 人种置信度 (0~1) |
| `raw_race_scores` | dict | 人种原始分数（6 类各自的概率） |

### 6.6 枚举映射表

**性别 (gender)**

| 值 | 标签 | F5 ODOT 枚举 |
|----|------|--------------|
| 0 | Female | `GENDER_FEMALE = 0` |
| 1 | Male | `GENDER_MALE = 1` |
| 2 | Unknown | `GENDER_UNKNOWN = 2` |

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

## 7. 运行效果示例

### 控制台输出

```
[INFO] 共找到 5 张图片
[INFO] 置信度阈值: 0.6
[INFO] 检测后端: opencv
[INFO] 输出文件: D:\F5MTL\person_scene_seg\test\gender_race_labels.json
------------------------------------------------------------
[1/5] celeb_28270.jpg ... 1张人脸 -> ['Male/Unknown']
[2/5] celeb_28271.jpg ... 1张人脸 -> ['Male/White']
[3/5] celeb_28272.jpg ... 1张人脸 -> ['Male/White']
[4/5] celeb_28273.jpg ... 1张人脸 -> ['Female/White']
[5/5] celeb_28274.jpg ... 1张人脸 -> ['Unknown/Unknown']
============================================================
[DONE] 标注完成!
  图片总数: 5
  人脸总数: 5
  性别 - 男性: 3, 女性: 1, 未知: 1
  人种 - 亚洲人: 0, 白人: 3, 中东人: 0, 印度人: 0, 拉丁裔: 0, 黑人: 0, 未知: 2
  无人脸图片: 0
  耗时: 227.8s (0.0 fps)
  输出: D:\F5MTL\person_scene_seg\test\gender_race_labels.json
```

> 注：首次运行耗时较长（下载模型权重），后续运行约 1~3 fps。

---

## 8. 常见问题

### Q: 某张图片标注为 Unknown？

说明模型对该图片的判断置信度低于阈值。可以：
- 降低阈值：`--conf 0.5`
- 换用更高精度的检测器：`--detector retinaface`
- 人工复查该图片的 `raw_gender_scores` / `raw_race_scores`

### Q: 一张图片检测到多个人脸？

工具会自动检测所有人脸，每个独立标注。`face_count` 显示数量，`faces` 数组包含每个人脸的详细信息。

### Q: 人种置信度普遍偏低？

这是正常的。人种分类本身难度较高，6 个类别之间的区分不如二分类（性别）明显。建议：
- 对人种使用较低阈值（如 `0.4`）或直接使用 `dominant_race` 而不做过滤
- 关注 `raw_race_scores` 中的概率分布，取 top-1 即可

### Q: 支持中文路径吗？

支持。图片读取使用 `np.fromfile` + `cv2.imdecode`，兼容中文路径。

### Q: 速度太慢怎么办？

- 使用 `opencv` 检测后端（最快）
- 安装 GPU 版 TensorFlow：`pip install tensorflow[and-cuda]`
- 分批处理，多机并行

### Q: 模型权重放在哪里？

权重位于工具目录下的 `.deepface/weights/`，脚本通过 `DEEPFACE_HOME` 环境变量自动指向该路径。无需手动配置。

### Q: `raw_race_scores` 中的 key 和 RACE_MAP 不一致？

deepface 返回的 key 是 `"latino hispanic"` 而非 `"latino"`，这不影响输出。`race` 字段已经过标准化映射（0~6），`raw_race_scores` 保留原始 key 供参考。

---

## 9. deepface 模型性能参考

| 指标 | 数值 |
|------|------|
| 性别分类准确率 | 97.44% |
| opencv 检测速度 | ~1-3 fps（CPU） |
| retinaface 检测速度 | ~0.3-0.5 fps（CPU） |
