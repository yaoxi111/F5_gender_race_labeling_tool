# F5 性别与人种自动标注工具 v2

基于 [DeepFace](https://github.com/serengil/deepface) + [YOLOv8](https://github.com/ultralytics/ultralytics) 的 **人体检测 + 人脸检测 + 性别分类 + 人种分类** 自动标注工具，输出与 F5 ODOT `FaceInfo` 结构对齐的 JSON 标注文件。

---

## 目录结构

```
F5_gender_race_labeling_tool/
├── label_gender_race.py        # 主脚本（本地版）
├── label_gender_race_v1.py     # v1 备份（仅人脸，无人体）
├── requirements.txt            # Python 依赖
├── setup_and_run.bat           # 一键安装依赖 + 运行
├── run_example.bat             # 示例运行脚本
├── TECHNICAL.md                # 技术文档
├── README.md                   # 本文档
├── yolov8n.pt                  # YOLOv8 人体检测模型 (6MB)
├── .deepface/weights/          # DeepFace 模型权重
│   ├── gender_model_weights.h5     # 性别分类 (512MB)
│   ├── race_model_single_batch.h5  # 人种分类 (512MB)
│   └── retinaface.h5               # RetinaFace 人脸检测 (113MB)
└── offline_package/            # 离线部署包（见第 10 节）
```

---

## 1. 环境安装

### 1.1 Python 要求

- Python 3.10 ~ 3.12（推荐 3.12）

### 1.2 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 1.3 模型权重

- `.deepface/weights/` 中的 3 个模型已内置，无需联网
- `yolov8n.pt` 首次运行时由 ultralytics 自动下载（约 6MB）

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
| `--detector` | `-d` | string | 否 | 人脸检测后端（推荐 `retinaface`） | `opencv` |
| `--viz-dir` | `-v` | string | 否 | 可视化输出目录（保存带框标注的图片） | — |

### 2.3 支持的图片格式

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`

工具会递归扫描输入文件夹下的所有子目录。

---

## 3. 使用示例

### 示例 1：推荐用法（retinaface + 可视化）

```bash
python label_gender_race.py -i D:\data\images -o D:\data\labels.json -v D:\data\viz -d retinaface
```

### 示例 2：大批量快速标注（opencv 最快）

```bash
python label_gender_race.py -i D:\data\images -o D:\data\labels.json -d opencv -c 0.5
```

### 示例 3：高精度精标（严格阈值）

```bash
python label_gender_race.py -i D:\data\images -o D:\data\labels.json -d retinaface -c 0.8
```

### 可视化颜色说明

- **绿色框** = 人体（person）
- **蓝色框** = 男性人脸
- **红色框** = 女性人脸
- **紫色框** = 未知性别人脸
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
1. 日常标注使用 `retinaface`（精度最高，人脸框最准确）
2. 大批量预标注（几千张以上）用 `opencv` 或 `mtcnn`
3. 人工抽查 20~50 张验证准确率

---

## 5. 置信度阈值说明

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| `0.5` | 宽松，大部分直接标注 | 数据量少，希望尽量多标注 |
| `0.6` | **默认推荐**，平衡精度和召回 | 通用场景 |
| `0.8` | 严格，只保留高置信度 | 对标注质量要求高 |
| `0.9` | 非常严格 | 只要几乎确定的结果 |

---

## 6. 输出格式

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
  "tool": "F5 Gender & Race Auto-Labeling Tool v2",
  "model": "deepface",
  "person_detector": "yolov8n",
  "face_detector_backend": "retinaface",
  "conf_threshold": 0.6,
  "total_images": 34,
  "total_persons": 35,
  "total_faces": 35,
  "errors": 0,
  "elapsed_seconds": 144.6,
  "fps": 0.23
}
```

### 6.3 labels（逐图标注）

```json
{
  "image_path": "celeb_28102.jpg",
  "image_abs_path": "D:\\...\\celeb_28102.jpg",
  "person_count": 1,
  "persons": [
    {
      "x1": 50, "y1": 30, "x2": 400, "y2": 600,
      "label": "person",
      "det_score": 0.9234
    }
  ],
  "face_count": 1,
  "faces": [
    {
      "x1": 101, "y1": 141, "x2": 539, "y2": 579,
      "det_score": 0.9908,
      "gender": 0, "gender_label": "Female", "gender_conf": 0.9908,
      "raw_gender_scores": {"Woman": 0.9908, "Man": 0.0092},
      "race": 1, "race_label": "White", "race_conf": 0.9912,
      "raw_race_scores": {"asian": 0.0, "white": 0.9912, ...}
    }
  ]
}
```

### 6.4 枚举映射表

**性别 (gender)**

| 值 | 标签 |
|----|------|
| 0 | Female |
| 1 | Male |
| 2 | Unknown |

**人种 (race)**

| 值 | 标签 | 含义 |
|----|------|------|
| 0 | Asian | 东亚/东南亚 |
| 1 | White | 白人 |
| 2 | Middle Eastern | 中东 |
| 3 | Indian | 南亚/印度 |
| 4 | Latino | 拉丁裔 |
| 5 | Black | 黑人 |
| 6 | Unknown | 未知 |

---

## 7. 大规模数据集保护

v2 包含以下保护机制：

1. **单图异常隔离** — 每张图片独立 try-catch，一张失败不影响后续
2. **定期检查点** — 每 50 张自动保存中间结果（`offline_package` 版支持自定义间隔）
3. **Ctrl+C 中断保护** — 中断时自动保存已处理结果
4. **内存释放** — 每张图处理后释放内存，防止 OOM
5. **错误统计** — 输出 JSON 中 `errors` 字段记录失败图片数

---

## 8. 常见问题

### Q: 某张图片标注为 Unknown？

置信度低于阈值。可降低阈值 `--conf 0.5` 或换用 `--detector retinaface`。

### Q: 检测到多个人体/人脸？

工具自动检测所有人体和人脸，每个独立标注。`person_count` / `face_count` 显示数量。

### Q: 人种置信度普遍偏低？

正常现象。建议对人种使用较低阈值，关注 `raw_race_scores` 概率分布。

### Q: 支持中文路径吗？

支持。使用 `np.fromfile` + `cv2.imdecode` 兼容中文路径。

### Q: 速度太慢？

- 用 `opencv` 后端（最快）
- 安装 GPU 版 TensorFlow：`pip install tensorflow[and-cuda]`
- 分批处理，多机并行

---

## 9. 模型性能参考

| 指标 | 数值 |
|------|------|
| 性别分类准确率 | ~97.44% |
| 人体检测（YOLOv8n） | ~1-2 fps |
| 人脸检测（opencv） | ~1-3 fps |
| 人脸检测（retinaface） | ~0.1-0.3 fps |

---

## 10. 离线部署

需要在无网络的服务器上使用？请参见 `offline_package/` 目录：

```bash
# 1. 拷贝 offline_package 文件夹到目标服务器
# 2. 从本地 .deepface/weights/ 拷贝 3 个模型到 offline_package/models/deepface/
# 3. 拷贝 yolov8n.pt 到 offline_package/models/
# 4. 安装依赖
pip install -r offline_package/requirements.txt
# 5. 运行
python offline_package/scripts/label_gender_race.py -i <图片文件夹> -o <输出.json> -d retinaface
```

离线版增强功能：
- 自定义检查点间隔：`--checkpoint 100`
- 启动时自动验证所有模型文件是否存在
- 支持 Windows（`install.bat`）和 Linux（`install.sh`）

详见 `offline_package/README.md`。
