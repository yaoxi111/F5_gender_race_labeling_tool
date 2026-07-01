# F5 性别与人种自动标注工具 v3

基于 [DeepFace](https://github.com/serengil/deepface) + [YOLOv8](https://github.com/ultralytics/ultralytics) 的 **人体检测 + 人脸检测 + 性别分类 + 人种分类** 自动标注工具。**v3 每张图输出一个独立 JSON 文件**，格式与 F5 MTL 训练管线对齐。支持自动检测并调用本地 GPU。

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
python label_gender_race.py --input <图片文件夹> --output <输出目录> [选项]
```

### 2.2 参数说明

| 参数 | 缩写 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|------|--------|
| `--input` | `-i` | string | **是** | 输入图片文件夹路径 | — |
| `--output` | `-o` | string | 否 | 输出目录路径（每张图一个JSON） | `./output_labels` |
| `--conf` | `-c` | float | 否 | 性别置信度阈值（低于阈值归为 male_or_gender_unknown） | `0.6` |
| `--race-conf` | — | float | 否 | 人种置信度阈值（人种模型置信度天然较低，建议 0.3） | `0.3` |
| `--race-argmax` | — | flag | 否 | 人种分类使用 argmax 模式（直接取最高分，完全消除 Unknown） | 关闭 |
| `--detector` | `-d` | string | 否 | 人脸检测后端 | `retinaface` |
| `--viz-dir` | `-v` | string | 否 | 可视化输出目录（保存带框标注的图片） | — |
| `--workers` | `-w` | int | 否 | 并行进程数（-1=自动使用所有 CPU 核心） | `1` |
| `--resize` | — | int | 否 | 图片缩放：长边最大像素数（0=不缩放） | `0` |

### 2.3 支持的图片格式

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`

工具会递归扫描输入文件夹下的所有子目录。

---

## 3. 使用示例

### 示例 1：推荐用法（retinaface + 输出目录）

```bash
python label_gender_race.py -i D:\data\images -o D:\data\labels -v D:\data\viz
```

### 示例 2：大批量快速标注（opencv 最快）

```bash
python label_gender_race.py -i D:\data\images -o D:\data\labels -d opencv -c 0.5
```

### 示例 3：完全消除人种 Unknown（argmax 模式）

```bash
python label_gender_race.py -i D:\data\images -o D:\data\labels --race-argmax
```

### 示例 4：高精度精标（严格阈值）

```bash
python label_gender_race.py -i D:\data\images -o D:\data\labels -c 0.8 --race-conf 0.5
```

### 可视化颜色说明

- **绿色框** = 人体（person）
- **蓝色框** = male_or_gender_unknown
- **红色框** = female
- 标签格式：`性别/人种`

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

## 5. 加速方案（10000+ 张图片）

### 5.1 时间估算（CPU 无 GPU）

| 后端 | 速度 | 10000 张耗时 | 说明 |
|------|------|-------------|------|
| `retinaface` | ~0.1 fps | ~28 小时 | 精度最高 |
| `mtcnn` | ~0.7 fps | ~4 小时 | 平衡 |
| `opencv` | ~2 fps | ~1.5 小时 | 最快 |

### 5.2 加速参数

**多进程并行**（`--workers`）：
```bash
# 使用 4 个进程并行处理（预计加速 2-3 倍）
python label_gender_race.py -i /data/images -o /data/labels -d opencv -w 4

# 自动使用所有 CPU 核心
python label_gender_race.py -i /data/images -o /data/labels -d opencv -w -1
```

**图片缩放**（`--resize`）：
```bash
# 将大图缩放到长边 1024px 再处理（大幅减少像素量，加速 2-4 倍）
python label_gender_race.py -i /data/images -o /data/labels -d opencv --resize 1024
```

**组合使用**（最大加速）：
```bash
# 4 进程 + 缩放 + opencv 后端 → 10000 张预计 20-30 分钟
python label_gender_race.py -i /data/images -o /data/labels -d opencv -w 4 --resize 1024
```

### 5.3 加速效果预估

| 配置 | 10000 张耗时 | 说明 |
|------|-------------|------|
| 单进程 + retinaface | ~28 小时 | 默认配置 |
| 单进程 + opencv | ~1.5 小时 | 换后端 |
| 4 进程 + opencv | ~30 分钟 | 多进程 |
| 4 进程 + opencv + resize 1024 | ~15 分钟 | 最大加速 |

---

## 6. 置信度阈值说明

### 6.1 性别阈值（`--conf`）

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| `0.5` | 宽松，大部分直接标注 | 数据量少，希望尽量多标注 |
| `0.6` | **默认推荐**，平衡精度和召回 | 通用场景 |
| `0.8` | 严格，只保留高置信度 | 对标注质量要求高 |
| `0.9` | 非常严格 | 只要几乎确定的结果 |

### 6.2 人种阈值（`--race-conf` / `--race-argmax`）

人种模型有 6 个类别，softmax 输出天然分散，置信度普遍低于性别模型。

| 模式 | 参数 | 效果 | 适用场景 |
|------|------|------|----------|
| 低阈值 | `--race-conf 0.3` | **默认**，仅过滤极低置信度 | 通用场景 |
| 中阈值 | `--race-conf 0.5` | 中等严格 | 对人种标注有一定质量要求 |
| argmax | `--race-argmax` | 直接取最高分，完全消除 Unknown | 需要每张脸都有人种标签 |

---

## 7. 输出格式（v3：每张图一个 JSON）

### 7.1 单图 JSON 格式

每张图片输出一个独立的 JSON 文件到指定目录：

```json
{
  "image_id": "000001",
  "image_file": "../images/000001.jpg",
  "mask_file": "../masks/000001.png",
  "width": 1920,
  "height": 1080,
  "persons": [
    {
      "id": 0,
      "bbox": [520, 180, 340, 580],
      "bbox_format": "xywh",
      "race": 0,
      "gender": 1
    }
  ]
}
```

> - `image_id` = 文件名去掉扩展名
> - `bbox` = `[x, y, width, height]`（左上角坐标 + 宽高）
> - `mask_file` 为占位路径，分割标注后续填充
> - 同时生成 `_summary.json` 汇总统计

### 7.2 性别枚举

| ID | 名称 | 说明 |
|----|------|------|
| 0 | female | 女性 |
| 1 | male_or_gender_unknown | 男性或未知（低于阈值时归为此类） |

### 7.3 人种枚举（暂不改动，后续调整）

| ID | 名称 |
|----|------|
| 0 | Asian |
| 1 | White |
| 2 | Middle Eastern |
| 3 | Indian |
| 4 | Latino |
| 5 | Black |
| 6 | Unknown |

### 7.4 GPU 支持

工具启动时自动检测本地 GPU：
- **TensorFlow GPU**：自动启用显存按需增长
- **PyTorch CUDA**：YOLOv8 自动调用
- 控制台会打印 GPU 设备信息

### 7.5 汇总文件 `_summary.json`

```json
{
  "metadata": {
    "tool": "F5 Gender & Race Auto-Labeling Tool v3",
    "total_images": 34,
    "total_persons": 35,
    "errors": 0,
    "elapsed_seconds": 144.6,
    "fps": 0.23
  },
  "statistics": {
    "total_persons": 35,
    "female": 12,
    "male_or_unknown": 23,
    "asian": 3,
    "white": 7,
    "unknown_race": 5
  }
}
```

---

## 8. 大规模数据集保护

v3 包含以下保护机制：

1. **单图异常隔离** — 每张图片独立 try-catch，一张失败不影响后续
2. **单图输出** — 每张图处理完立即写入独立 JSON，崩溃不丢已处理结果
3. **Ctrl+C 中断保护** — 中断时已处理的图片 JSON 已落盘
4. **内存释放** — 每张图处理后释放内存，防止 OOM
5. **错误统计** — 汇总文件 `_summary.json` 中记录失败图片数

---

## 9. 常见问题

### Q: 某张图片标注为 Unknown？

性别：低于阈值时归为 `male_or_gender_unknown`（ID=1），不会出现 Unknown。
人种：置信度低于阈值标记为 Unknown。可降低阈值 `--race-conf 0.3` 或使用 `--race-argmax` 完全消除。

### Q: 检测到多个人体/人脸？

工具自动检测所有人体和人脸，通过中心点距离匹配后输出统一的 `persons` 数组。每个人有独立的 `id`、`bbox`、`race`、`gender`。

### Q: 人种置信度普遍偏低？

正常现象。人种模型有 6 个类别，softmax 输出天然分散。解决方案：
1. **降低人种阈值**：`--race-conf 0.3`（默认已是 0.3）
2. **argmax 模式**：`--race-argmax` 直接取最高分，完全消除 Unknown

### Q: 支持中文路径吗？

支持。使用 `np.fromfile` + `cv2.imdecode` 兼容中文路径。

### Q: 速度太慢？

- 用 `opencv` 后端（最快）
- 安装 GPU 版 TensorFlow：`pip install tensorflow[and-cuda]`（工具会自动检测并调用 GPU）
- 使用 `--workers` 多进程并行
- 使用 `--resize` 缩放图片
- 分批处理，多机并行

---

## 10. 模型性能参考

| 指标 | 数值 |
|------|------|
| 性别分类准确率 | ~97.44% |
| 人体检测（YOLOv8n） | ~1-2 fps |
| 人脸检测（opencv） | ~1-3 fps |
| 人脸检测（retinaface） | ~0.1-0.3 fps |

---

## 11. 离线部署

需要在无网络的服务器上使用？请参见 `offline_package/` 目录：

```bash
# 1. 拷贝 offline_package 文件夹到目标服务器
# 2. 从本地 .deepface/weights/ 拷贝 3 个模型到 offline_package/models/deepface/
# 3. 拷贝 yolov8n.pt 到 offline_package/models/
# 4. 安装依赖
pip install -r offline_package/requirements.txt
# 5. 运行
python offline_package/scripts/label_gender_race.py -i <图片文件夹> -o <输出目录> -d retinaface
```

离线版增强功能：
- 每张图输出独立 JSON 文件，崩溃不丢已处理结果
- 自定义检查点间隔：`--checkpoint 100`
- 启动时自动验证所有模型文件是否存在
- 自动检测并调用本地 GPU
- 支持 Windows（`install.bat`）和 Linux（`install.sh`）

详见 `offline_package/README.md`。

---

## 12. GUI 图形界面工具

v3 新增 `label_gui.py` 图形界面，无需命令行即可操作。

### 启动

```bash
python label_gui.py
```

### 功能

- **路径设置**：输入目录、输出目录、可视化目录（输出目录不存在自动创建）
- **标注内容开关**：人体检测 / 人脸检测 / 性别分类 / 人种分类，可独立开关
- **精度预设**：高精度 / 均衡 / 快速 / argmax 四种模式一键切换
- **参数调节**：检测后端、性别阈值、人种阈值、并行进程数
- **标注预览**：完成后可翻页浏览可视化结果

### 依赖

```bash
pip install Pillow
```
