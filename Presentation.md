# F5 自动标注工具 v3 — 演讲稿

> 时长：约 10-15 分钟
> 配合 PPT 或直接展示工具运行效果
> 参考文档：[飞书 Wiki](https://mi.feishu.cn/wiki/UkLDw5Lcnitn1ukpGsHcClMZnMf)

---

## 一、开场（1 分钟）

大家好，今天给大家介绍我们的 **F5 性别与人种自动标注工具 v2**。

这个工具解决的核心问题是：在 F5 人物场景分割项目中，我们需要对大量图片中的人体和人脸进行属性标注——包括人体位置、性别、人种。如果纯人工标注，一张图可能需要几分钟，几万张图就是巨大的工作量。

**这个工具可以做到：自动检测人体 + 自动检测人脸 + 自动分类性别和人种，一键输出标准 JSON 标注文件。**

先说一下我们目前的测试现状：
1. **图片中脸部占比大的场景效果比较好**，性别准确率 90% 以上，人种在 80% 左右
2. **脸部在图片中占比较少的场景效果一般**，性别准确率 80% 左右，人种在 50% 左右
3. **人体标注准确率 98% 以上**，基本都可以标注成功且比较准确

---

## 二、技术架构（3 分钟）

### 2.1 整体流程

工具的处理流程分为 **三个阶段**：

```
输入图片 → 人体检测 → 人脸检测 + 分类 → 输出 JSON + 可视化图片
```

**第一阶段：人体检测**
- 使用 **YOLOv8 Nano** 模型
- 这是 Ultralytics 推出的目标检测模型，在 COCO 数据集上预训练
- 我们只使用其中的 "person" 类别，检测图片中的所有人体区域
- 模型只有 6MB，速度快

**第二阶段：人脸检测 + 属性分类**
- 使用 **DeepFace** 框架，一步完成三件事：
  1. **人脸检测**：定位人脸位置（推荐 RetinaFace 后端，精度最高）
  2. **人脸对齐**：通过仿射变换矫正旋转，确保人脸正面朝上
  3. **属性分类**：用 VGG 模型做性别二分类（~97% 准确率）和人种六分类（~90% 准确率）

**第三阶段：输出**
- JSON 标注文件：每张图包含人体框坐标、人脸框坐标、性别、人种、置信度
- 可视化图片：绿色框标人体，蓝/红/紫框标人脸

### 2.2 多尺度检测 Fallback（v2 新增）

为了让小人脸也能被检测到，v2 版本实现了 **4 轮回退策略**：

| 轮次 | 策略 | 说明 |
|------|------|------|
| 第 1 轮 | 原图 + 选定后端 | 默认检测 |
| 第 2 轮 | 1.5x 放大 + 同一后端 | 小人脸放大后再检测 |
| 第 3 轮 | 原图 + RetinaFace | 换用最高精度后端 |
| 第 4 轮 | 原图 + MTCNN | 仅当后端为 opencv 时触发 |

**只有当前一轮没有检测到有效人脸时，才会触发下一轮。** 这样既保证了检测率，又不会过度增加耗时。

### 2.3 技术选型理由

| 组件 | 为什么选它 |
|------|-----------|
| YOLOv8 | 人体检测精度高，模型小（6MB），速度快 |
| DeepFace | 封装了多种人脸分析模型，API 简洁，支持 7 种检测后端 |
| RetinaFace | 人脸检测精度最高，支持 5 个面部关键点定位 |
| VGG-Face | 经典的人脸识别架构，在性别/人种分类上准确率高 |

---

## 三、演示（3-4 分钟）

### 3.1 基本用法演示

```bash
# 推荐用法：retinaface + 人种独立阈值
python label_gender_race.py -i test2 -o test2/labels.json -v test2/viz

# 完全消除人种 Unknown（argmax 模式）
python label_gender_race.py -i test2 -o test2/labels.json --race-argmax -v test2/viz
```

展示：
1. 控制台输出：逐张显示检测结果
2. 输出 JSON：打开 labels.json，展示结构
3. 可视化图片：打开 viz 目录，展示带框标注的图片

### 3.2 效果展示

展示几个典型场景：
- **单人正面照**：人体框 + 人脸框 + 正确的性别/人种分类
- **多人场景**：每个人体和人脸都被检测到
- **低置信度案例**：展示 Unknown 标注的效果

### 3.3 关键数据

| 指标 | 大脸场景 | 小脸场景 |
|------|---------|---------|
| 人体检测率 | 98%+ | 98%+ |
| 性别准确率 | 90%+ | ~80% |
| 人种准确率 | ~80% | ~50% |

---

## 四、人种分类策略（2 分钟）

人种模型有 6 个类别（Asian / White / Middle Eastern / Indian / Latino / Black），softmax 输出天然分散，置信度普遍低于性别模型。我们提供了三种策略：

### 4.1 独立阈值模式（默认）

```bash
python label_gender_race.py -i images -o labels.json --race-conf 0.3
```

- 性别和人种使用**独立的置信度阈值**
- 性别阈值 `--conf` 默认 0.6，人种阈值 `--race-conf` 默认 0.3
- 低于阈值的标记为 Unknown

### 4.2 Argmax 模式（完全消除 Unknown）

```bash
python label_gender_race.py -i images -o labels.json --race-argmax
```

- 直接取最高分的类别，不做阈值过滤
- **完全消除人种 Unknown**
- 适合需要每张脸都有人种标签的场景

### 4.3 选择建议

| 模式 | 参数 | 效果 | 适用场景 |
|------|------|------|----------|
| 低阈值 | `--race-conf 0.3` | 默认，仅过滤极低置信度 | 通用场景 |
| 中阈值 | `--race-conf 0.5` | 中等严格 | 对人种标注有一定质量要求 |
| argmax | `--race-argmax` | 直接取最高分，完全消除 Unknown | 需要每张脸都有人种标签 |

---

## 五、大规模数据集保护（2 分钟）

当处理几万甚至几十万张图片时，我们最担心的是：跑到一半崩了，之前的结果全丢了。

v2 版本针对这个问题设计了 **五重保护机制**：

| 保护机制 | 说明 |
|----------|------|
| **单图异常隔离** | 每张图独立 try-catch，一张崩了不影响后续图片 |
| **定期检查点** | 每 50 张自动保存中间结果到 JSON 文件 |
| **Ctrl+C 中断保护** | 用户按 Ctrl+C 时，自动保存已处理的结果 |
| **内存释放** | 每张图处理后释放内存，防止内存溢出 |
| **错误统计** | 输出 JSON 中记录失败图片数量 |

**即使跑到第 9999 张时崩溃了，前 9950 张的结果也已经保存在输出文件中。**

---

## 六、加速方案（1 分钟）

处理 10000+ 张图片时，可以组合使用以下加速手段：

### 6.1 多进程并行（`--workers`）

```bash
# 使用 4 个进程并行处理（预计加速 2-3 倍）
python label_gender_race.py -i /data/images -o /data/labels.json -d opencv -w 4

# 自动使用所有 CPU 核心
python label_gender_race.py -i /data/images -o /data/labels.json -d opencv -w -1
```

### 6.2 图片缩放（`--resize`）

```bash
# 将大图缩放到长边 1024px 再处理（加速 2-4 倍）
python label_gender_race.py -i /data/images -o /data/labels.json -d opencv --resize 1024
```

### 6.3 组合使用（最大加速）

```bash
# 4 进程 + 缩放 + opencv 后端 → 10000 张预计 20-30 分钟
python label_gender_race.py -i /data/images -o /data/labels.json -d opencv -w 4 --resize 1024
```

---

## 七、离线部署（1 分钟）

很多实际场景中，标注服务器是内网环境，无法联网下载模型和依赖包。

我们提供了 **完全离线的部署包**，包含：

```
offline_package/                    总大小：约 1.8 GB
├── scripts/                        主程序
├── packages/                       80+ 个 pip 离线安装包
├── models/                         4 个 AI 模型权重
│   ├── yolov8n.pt                  6MB（人体检测）
│   ├── gender_model_weights.h5     512MB（性别分类）
│   ├── race_model_single_batch.h5  512MB（人种分类）
│   └── retinaface.h5               113MB（人脸检测）
├── install.bat / install.sh        一键安装脚本
└── README.md                       详细部署教程
```

**部署三步走**：
1. U 盘拷贝到目标服务器
2. 运行 `install.bat`（或 `pip install --no-index --find-links=packages -r requirements.txt`）
3. 开始标注

**不需要联网，不需要额外下载任何东西。**

---

## 八、输出格式说明（1 分钟）

输出的 JSON 文件结构：

```json
{
  "metadata": {
    "tool": "F5 Gender & Race Auto-Labeling Tool v2",
    "model": "deepface",
    "person_detector": "yolov8n",
    "face_detector_backend": "retinaface",
    "conf_threshold": 0.6,
    "race_conf_threshold": 0.3,
    "total_images": 34,
    "total_persons": 35,
    "total_faces": 35,
    "errors": 0
  },
  "statistics": { "male": 5, "female": 12, "asian": 3, "white": 7, ... },
  "labels": [
    {
      "image_path": "photo001.jpg",
      "person_count": 1,
      "persons": [{"x1": 50, "y1": 30, "x2": 400, "y2": 600, "label": "person"}],
      "face_count": 1,
      "faces": [{
        "x1": 101, "y1": 141, "x2": 539, "y2": 579,
        "gender_label": "Female", "gender_conf": 0.99,
        "race_label": "White", "race_conf": 0.95,
        "raw_gender_scores": {"Woman": 0.99, "Man": 0.01},
        "raw_race_scores": {"asian": 0.0, "white": 0.95, ...}
      }]
    }
  ]
}
```

> 注：`race_conf_threshold` 显示为 `"argmax"` 表示使用了 argmax 模式。

每张图包含：
- **persons**：人体框坐标（YOLOv8 检测）
- **faces**：人脸框坐标 + 性别 + 人种 + 置信度
- **raw_gender_scores / raw_race_scores**：原始概率分布，方便下游灵活使用

---

## 九、使用建议（1 分钟）

### 检测后端选择

| 场景 | 推荐后端 | 理由 |
|------|---------|------|
| 大批量预标注（几千张+） | `opencv` | 速度最快 |
| 日常标注 | `mtcnn` | 精度/速度平衡 |
| 精标/小批量 | `retinaface` | 精度最高（默认） |

### 推荐工作流

1. 先用 `opencv` + `--resize 1024` 快速跑一遍全部数据
2. 对 Unknown 样本用 `retinaface` 精标
3. 人种 Unknown 可用 `--race-argmax` 完全消除
4. 人工抽查 20-50 张验证准确率

---

## 十、总结（1 分钟）

**F5 自动标注工具 v2 的核心优势**：

1. **全自动**：人体检测 + 人脸检测 + 性别/人种分类，一键完成
2. **高精度**：YOLOv8 人体检测 + RetinaFace 人脸检测 + VGG 分类模型
3. **智能回退**：4 轮多尺度检测 fallback，小人脸也能检测到
4. **灵活配置**：性别/人种独立阈值 + argmax 模式，适应不同场景
5. **稳定可靠**：五重崩溃保护，大规模数据集不丢结果
6. **完全离线**：1.8GB 部署包，拷贝即用，无需联网
7. **标准输出**：JSON 格式与 F5 ODOT 对齐，直接对接训练管线

**GitHub 开源地址**：https://github.com/yaoxi111/F5_gender_race_labeling_tool

谢谢大家！

---

## 附：常见提问准备

**Q: 人种分类准确率只有 90%，够用吗？**
A: 人种分类本身难度较高，6 类之间容易混淆。我们提供了 `raw_race_scores` 原始概率分布，下游训练时可以利用这些软标签，而不是只用硬分类结果。另外可以用 `--race-argmax` 完全消除 Unknown。

**Q: 能不能训练自定义模型？**
A: 当前版本使用预训练模型。如果需要针对特定场景微调，可以收集标注数据后对 VGG 模型进行 fine-tune，这是后续优化方向。

**Q: 处理速度能更快吗？**
A: 三种方式：①用 `opencv` 后端；②用 `--workers` 多进程并行；③用 `--resize` 缩放图片。组合使用 10000 张预计 20-30 分钟。

**Q: 支持视频标注吗？**
A: 当前版本只支持图片。视频可以先抽帧再标注，后续可以考虑直接支持视频输入。

**Q: 小人脸检测不到怎么办？**
A: v2 已内置 4 轮多尺度 fallback（原图→1.5x 放大→RetinaFace→MTCNN），会自动尝试多种策略。如果仍然检测不到，可以手动放大图片后单独处理。
