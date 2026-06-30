"""
F5 性别与人种自动标注工具 v3 (deepface 版)
人体检测 + 人脸检测 + 性别分类 + 人种分类: deepface + YOLOv8
每张图输出一个独立 JSON 标注文件，格式与 F5 MTL 训练管线对齐。

用法:
    python label_gender_race.py -i <图片文件夹> -o <输出目录> [-c 0.6] [--race-conf 0.3] [-d retinaface] [-w 4] [--resize 1024]

依赖:
    pip install deepface opencv-python tf-keras ultralytics
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# 指向本地 weights 目录，避免每次从 GitHub 下载
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DEEPFACE_HOME"] = _SCRIPT_DIR

# 解决 DeepFace 与 Keras 3.x 的兼容性问题
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# TensorFlow CPU 线程优化（默认 4 线程，可被 --workers 覆盖）
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")

# 修复 Windows GBK 编码
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ─────────── GPU 检测 ───────────

def _detect_gpu():
    """检测可用 GPU 并配置 TensorFlow。返回 GPU 信息字符串。"""
    gpu_info = "未检测到 GPU"
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_info = f"TensorFlow GPU: {len(gpus)} 个设备"
            gpu_names = [g.name for g in gpus]
            print(f"[INFO] 检测到 GPU: {gpu_names}")
        else:
            print("[INFO] 未检测到 TensorFlow GPU，使用 CPU")
    except Exception as e:
        print(f"[INFO] GPU 检测跳过: {e}")

    # 检测 PyTorch GPU（YOLOv8 会自动使用）
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info += f" | PyTorch CUDA: {gpu_name}"
            print(f"[INFO] PyTorch CUDA 可用: {gpu_name}")
        else:
            print("[INFO] PyTorch CUDA 不可用")
    except ImportError:
        pass

    return gpu_info

_GPU_INFO = None


def _get_gpu_info():
    global _GPU_INFO
    if _GPU_INFO is None:
        _GPU_INFO = _detect_gpu()
    return _GPU_INFO


# ─────────── 常量 ───────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# 绘图颜色 (BGR)
COLOR_PERSON = (0, 255, 0)       # 绿色 - 人体
COLOR_MALE = (255, 128, 0)       # 蓝色 - 男性
COLOR_FEMALE = (0, 80, 255)      # 红色 - 女性
COLOR_UNKNOWN_G = (180, 0, 180)  # 紫色 - 未知性别

# 性别映射: deepface 返回 "Woman"/"Man" -> 0=female, 1=male_or_gender_unknown
# 低于阈值时归为 1（male_or_gender_unknown）
GENDER_MAP = {"Woman": 0, "Man": 1}
GENDER_LABELS = ["female", "male_or_gender_unknown"]

# 人种映射: deepface 返回 6 类 -> 合并为 4 类
# 0=yellow(Asian+Indian), 1=white(White+Middle Eastern), 2=brown(Black+Latino), 3=race_unknown
RACE_MAP_DEEPFACE = {
    "asian": 0,           # -> yellow
    "white": 1,           # -> white
    "middle eastern": 1,  # -> white
    "indian": 0,          # -> yellow
    "latino": 2,          # -> brown
    "black": 2,           # -> brown
}
RACE_LABELS = ["yellow", "white", "brown", "race_unknown"]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ─────────── 人体检测 ───────────

# YOLOv8 模型（懒加载）
_yolo_model = None

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")  # nano 版本，约 6MB，首次自动下载
    return _yolo_model


def detect_persons(img, conf_threshold=0.5):
    """
    使用 YOLOv8 检测人体（COCO class 0 = person）。
    返回 [(x1, y1, x2, y2, score), ...] 列表。
    """
    model = _get_yolo_model()
    results = model(img, conf=conf_threshold, verbose=False, classes=[0])  # 只检测 person 类

    persons = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            score = float(box.conf[0])
            persons.append((int(x1), int(y1), int(x2), int(y2), score))

    return persons


# ─────────── 核心分析 ───────────

def _classify_race(race_scores, race_conf_threshold, race_argmax):
    """
    根据人种分数进行分类。
    race_argmax=True 时直接取最高分；否则使用独立阈值。
    返回 (race_id, race_conf)。
    """
    if not race_scores:
        return 3, 0.0  # race_unknown

    # 找到最高分的类别
    max_label = max(race_scores, key=race_scores.get)
    max_score = float(race_scores[max_label]) / 100.0

    if race_argmax:
        # argmax 模式：直接取最高分，不设阈值
        return RACE_MAP_DEEPFACE.get(max_label.lower(), 3), max_score

    # 阈值模式：使用人种独立阈值
    if max_score < race_conf_threshold:
        return 3, max_score  # race_unknown
    return RACE_MAP_DEEPFACE.get(max_label.lower(), 3), max_score


def _extract_face_info(r, conf_threshold, race_conf_threshold, race_argmax, w_img, h_img):
    """
    从 DeepFace 结果中提取单张人脸信息。
    返回 dict 或 None（被过滤时）。
    """
    region = r.get("region", {})
    x = region.get("x", 0)
    y = region.get("y", 0)
    w = region.get("w", 0)
    h = region.get("h", 0)

    # 过滤掉过小的人脸（宽或高 < 30 像素）
    if w < 30 or h < 30:
        return None

    # 过滤掉过大的"人脸"（覆盖图片 70% 以上 = 检测失败，返回了整张图）
    if w > w_img * 0.7 and h > h_img * 0.7:
        return None

    # ── 性别 ──
    gender_label = r.get("dominant_gender", "Unknown")
    gender_scores = r.get("gender", {})
    raw_gender_conf = max(gender_scores.values()) if gender_scores else 0.0
    gender_conf = float(raw_gender_conf) / 100.0

    # 0=female, 1=male_or_gender_unknown（低于阈值也归为 1）
    if gender_label == "Woman" and gender_conf >= conf_threshold:
        gender_id = 0  # female
    else:
        gender_id = 1  # male_or_gender_unknown

    # ── 人种（使用独立阈值或 argmax）──
    race_scores = r.get("race", {})
    race_id, race_conf = _classify_race(race_scores, race_conf_threshold, race_argmax)

    return {
        "bbox": [x, y, w, h],
        "bbox_format": "xywh",
        "det_score": round(gender_conf, 4),
        # 性别
        "gender": gender_id,
        "gender_label": GENDER_LABELS[gender_id],
        "gender_conf": round(gender_conf, 4),
        "raw_gender_scores": {
            k: round(float(v) / 100.0, 4) for k, v in gender_scores.items()
        },
        # 人种
        "race": race_id,
        "race_label": RACE_LABELS[race_id],
        "race_conf": round(race_conf, 4),
        "raw_race_scores": {
            k: round(float(v) / 100.0, 4) for k, v in race_scores.items()
        },
    }


def _try_deepface(img, detector_backend, enforce=False):
    """调用 DeepFace.analyze，返回结果列表或空列表。"""
    from deepface import DeepFace
    try:
        return DeepFace.analyze(
            img_path=img,
            actions=["gender", "race"],
            detector_backend=detector_backend,
            enforce_detection=enforce,
            silent=True,
        )
    except Exception:
        return []


def analyze_image(img, detector_backend, conf_threshold,
                  race_conf_threshold=0.3, race_argmax=False):
    """
    对单张图片做人脸检测 + 性别分类 + 人种分类。
    使用 DeepFace.analyze() 一步完成检测 + 分类，确保分类精度。
    支持多尺度检测 fallback：原图 → 放大 1.5x → retinaface。
    返回 FaceInfo 列表。
    """
    h_img, w_img = img.shape[:2]

    # ── 第 1 轮：原图检测 ──
    demographies = _try_deepface(img, detector_backend)
    face_infos = []
    for r in demographies:
        info = _extract_face_info(r, conf_threshold, race_conf_threshold, race_argmax, w_img, h_img)
        if info is not None:
            face_infos.append(info)

    # ── 第 2 轮：如果原图无有效人脸，尝试放大 1.5x 重试 ──
    if not face_infos:
        scale_up = 1.5
        img_up = cv2.resize(img, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC)
        h_up, w_up = img_up.shape[:2]
        demographies = _try_deepface(img_up, detector_backend)
        for r in demographies:
            info = _extract_face_info(r, conf_threshold, race_conf_threshold, race_argmax, w_up, h_up)
            if info is not None:
                # 还原坐标到原图 (bbox: [x, y, w, h])
                bx, by, bw, bh = info["bbox"]
                info["bbox"] = [int(bx / scale_up), int(by / scale_up),
                                int(bw / scale_up), int(bh / scale_up)]
                face_infos.append(info)
        del img_up

    # ── 第 3 轮：如果仍无结果且不是 retinaface，用 retinaface 最终尝试 ──
    if not face_infos and detector_backend != "retinaface":
        demographies = _try_deepface(img, "retinaface")
        for r in demographies:
            info = _extract_face_info(r, conf_threshold, race_conf_threshold, race_argmax, h_img, w_img)
            if info is not None:
                face_infos.append(info)

    # ── 第 4 轮：如果 opencv 失败且不是 mtcnn，用 mtcnn 重试 ──
    # （保留原有逻辑，但移到多尺度检测之后）
    if not face_infos and detector_backend == "opencv":
        demographies = _try_deepface(img, "mtcnn")
        for r in demographies:
            info = _extract_face_info(r, conf_threshold, race_conf_threshold, race_argmax, h_img, w_img)
            if info is not None:
                face_infos.append(info)

    return face_infos


# ─────────── 可视化绘制 ───────────

def draw_results(img, face_infos, person_infos):
    """
    在图片上绘制人体框（绿色）和人脸框（蓝/红/紫）。
    返回绘制后的图片副本。
    """
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # ── 先画人体框（绿色，最底层）──
    for p in person_infos:
        if "bbox" in p:
            bx, by, bw, bh = p["bbox"]
            px1, py1, px2, py2 = bx, by, bx + bw, by + bh
        else:
            px1, py1, px2, py2 = p["x1"], p["y1"], p["x2"], p["y2"]
        cv2.rectangle(vis, (px1, py1), (px2, py2), COLOR_PERSON, 3)

        # 标签: "person"
        label = "person"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y1 = py1 - th - 12
        if label_y1 < 0:
            label_y1 = py1 + 4
        cv2.rectangle(vis, (px1, label_y1), (px1 + tw + 10, label_y1 + th + 10), COLOR_PERSON, -1)
        cv2.putText(vis, label, (px1 + 5, label_y1 + th + 4), font, font_scale, (255, 255, 255), thickness)

    # ── 再画人脸框（上层）──
    for face in face_infos:
        bx, by, bw, bh = face["bbox"]
        x1, y1 = bx, by
        x2, y2 = bx + bw, by + bh

        # 性别决定框颜色
        gid = face["gender"]
        if gid == 0:
            color = COLOR_FEMALE
        else:
            color = COLOR_MALE

        # 画人脸框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        # 标签文字: "Male/White 98%"
        g_label = face["gender_label"]
        r_label = face["race_label"]
        g_conf = face["gender_conf"]
        r_conf = face["race_conf"]
        line1 = f"{g_label}/{r_label}"
        line2 = f"{g_conf*100:.0f}%/{r_conf*100:.0f}%"

        (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        box_w = max(tw1, tw2) + 10
        box_h = th1 + th2 + 16

        label_y1 = y1 - box_h - 4
        if label_y1 < 0:
            label_y1 = y1 + 4
        label_y2 = label_y1 + box_h

        cv2.rectangle(vis, (x1, label_y1), (x1 + box_w, label_y2), color, -1)
        cv2.putText(vis, line1, (x1 + 5, label_y1 + th1 + 4), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis, line2, (x1 + 5, label_y1 + th1 + th2 + 12), font, font_scale, (255, 255, 255), thickness)

    return vis


# ─────────── 图片缩放 ───────────

def resize_image(img, max_side):
    """
    将图片等比缩放，使长边不超过 max_side 像素。
    返回 (缩放后的图片, 缩放比例)。
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def scale_boxes(boxes, scale):
    """将框坐标按比例还原到原图尺寸。支持 dict(bbox:[x,y,w,h]) 和 tuple(x1,y1,x2,y2,score)。"""
    if scale == 1.0:
        return boxes
    result = []
    for box in boxes:
        if isinstance(box, dict) and "bbox" in box:
            b = dict(box)
            bx, by, bw, bh = b["bbox"]
            b["bbox"] = [int(bx / scale), int(by / scale), int(bw / scale), int(bh / scale)]
            result.append(b)
        elif isinstance(box, dict):
            b = dict(box)
            for k in ("x1", "y1", "x2", "y2"):
                if k in b:
                    b[k] = int(b[k] / scale)
            result.append(b)
        elif isinstance(box, (list, tuple)):
            result.append(tuple(int(v / scale) for v in box))
        else:
            result.append(box)
    return result


# ─────────── 单图处理（可被多进程调用）────────────

def _match_persons_faces(person_boxes, face_infos):
    """
    将人体检测框和人脸检测结果匹配，输出统一的 persons 数组。
    每个 person 有 bbox(xywh) + race + gender。
    """
    result = []
    used_faces = set()

    for idx, (px1, py1, px2, py2, pscore) in enumerate(person_boxes):
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        best_face = None
        best_dist = float('inf')

        for fi, face in enumerate(face_infos):
            if fi in used_faces:
                continue
            bx, by, bw, bh = face["bbox"]
            fcx, fcy = bx + bw / 2, by + bh / 2
            dist = ((pcx - fcx) ** 2 + (pcy - fcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_face = fi

        pw, ph = px2 - px1, py2 - py1
        if best_face is not None and best_dist < max(pw, ph) * 0.8:
            used_faces.add(best_face)
            face = face_infos[best_face]
            result.append({
                "id": idx,
                "bbox": [int(px1), int(py1), int(pw), int(ph)],
                "bbox_format": "xywh",
                "race": face["race"],
                "gender": face["gender"],
                "gender_conf": face.get("gender_conf", 0),
                "race_conf": face.get("race_conf", 0),
            })
        else:
            result.append({
                "id": idx,
                "bbox": [int(px1), int(py1), int(pw), int(ph)],
                "bbox_format": "xywh",
                "race": 3,   # race_unknown
                "gender": 1, # male_or_gender_unknown
                "gender_conf": 0,
                "race_conf": 0,
            })

    # 未匹配到人体框的人脸，用其 bbox 作为 person bbox
    for fi, face in enumerate(face_infos):
        if fi in used_faces:
            continue
        bx, by, bw, bh = face["bbox"]
        result.append({
            "id": len(result),
            "bbox": [int(bx), int(by), int(bw), int(bh)],
            "bbox_format": "xywh",
            "race": face["race"],
            "gender": face["gender"],
            "gender_conf": face.get("gender_conf", 0),
            "race_conf": face.get("race_conf", 0),
        })

    return result


def _process_single_image(args):
    """
    处理单张图片，返回 (rel_path, abs_path, persons_output, error_msg)。
    用于多进程模式。
    """
    img_path, input_dir, detector_backend, conf_threshold, max_resize, race_conf_threshold, race_argmax = args
    rel_path = os.path.relpath(img_path, input_dir)

    try:
        raw = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return (rel_path, os.path.abspath(img_path), [], "无法读取")

        h_orig, w_orig = img.shape[:2]

        # 图片缩放
        scale = 1.0
        if max_resize and max_resize > 0:
            img, scale = resize_image(img, max_resize)

        # 人体检测
        person_boxes = detect_persons(img)

        # 人脸检测 + 分类
        face_infos = analyze_image(img, detector_backend, conf_threshold,
                                   race_conf_threshold, race_argmax)

        # 还原到原图坐标
        if scale != 1.0:
            person_boxes = [(int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale), s)
                           for (x1, y1, x2, y2, s) in person_boxes]
            face_infos = scale_boxes(face_infos, scale)

        # 匹配人体和人脸，输出统一 persons 数组
        persons_output = _match_persons_faces(person_boxes, face_infos)

        del img, raw
        return (rel_path, os.path.abspath(img_path), persons_output, w_orig, h_orig, None)

    except Exception as e:
        return (rel_path, os.path.abspath(img_path), [], 0, 0, str(e))


# ─────────── 主流程 ───────────

def collect_images(input_dir, skip_dirs=None):
    skip = set(skip_dirs) if skip_dirs else set()
    images = []
    for root, dirs, files in os.walk(input_dir):
        # 排除指定目录
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                images.append(os.path.join(root, f))
    images.sort()
    return images


def _update_stats(stats, persons_output):
    """更新统计信息。persons_output 是统一的 persons 数组。"""
    stats["total_images"] += 1
    stats["total_persons"] += len(persons_output)
    if not persons_output:
        stats["no_person"] += 1
    for p in persons_output:
        g = p["gender"]
        if g == 0: stats["female"] += 1
        else: stats["male_or_unknown"] += 1
        rid = p["race"]
        if rid == 0: stats["yellow"] += 1
        elif rid == 1: stats["white"] += 1
        elif rid == 2: stats["brown"] += 1
        else: stats["race_unknown"] += 1


def _save_single_json(output_dir, image_id, rel_path, w, h, persons_output):
    """保存单张图的标注 JSON 文件。"""
    data = {
        "image_id": image_id,
        "image_file": rel_path,
        "mask_file": f"../masks/{image_id}.png",
        "width": w,
        "height": h,
        "persons": persons_output,
    }
    out_path = os.path.join(output_dir, f"{image_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    return out_path


def run_labeling(input_dir, output_dir, conf_threshold, detector_backend,
                 viz_dir=None, workers=1, max_resize=0,
                 race_conf_threshold=0.3, race_argmax=False):
    skip = {"viz", "labels", "output"}
    images = collect_images(input_dir, skip_dirs=skip)
    if not images:
        print(f"[ERROR] 未找到图片: {input_dir}")
        sys.exit(1)

    # 自动检测 CPU 核心数
    if workers == -1:
        workers = os.cpu_count() or 4

    # 初始化 GPU 检测
    gpu_info = _get_gpu_info()

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 共找到 {len(images)} 张图片")
    print(f"[INFO] 性别置信度阈值: {conf_threshold}")
    if race_argmax:
        print(f"[INFO] 人种分类: argmax 模式（直接取最高分）")
    else:
        print(f"[INFO] 人种置信度阈值: {race_conf_threshold}")
    print(f"[INFO] 检测后端: {detector_backend}")
    print(f"[INFO] 并行进程数: {workers}")
    print(f"[INFO] GPU: {gpu_info}")
    if max_resize > 0:
        print(f"[INFO] 图片缩放: 长边不超过 {max_resize}px")
    print(f"[INFO] 输出目录: {output_dir}")
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[INFO] 可视化输出: {viz_dir}")
    print("-" * 60)

    stats = {
        "total_images": 0, "total_persons": 0,
        "female": 0, "male_or_unknown": 0,
        "yellow": 0, "white": 0, "brown": 0, "race_unknown": 0,
        "no_person": 0, "errors": 0,
    }
    start_time = time.time()
    total = len(images)

    if workers > 1:
        # ── 多进程模式 ──
        import multiprocessing as mp

        task_args = [
            (img_path, input_dir, detector_backend, conf_threshold, max_resize,
             race_conf_threshold, race_argmax)
            for img_path in images
        ]

        print(f"[INFO] 启动 {workers} 个进程并行处理...")
        all_viz_data = []
        with mp.Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_process_single_image, task_args)):
                rel_path, abs_path, persons_output, w, h, error = result
                image_id = Path(rel_path).stem

                if error:
                    print(f"[{i + 1}/{total}] {rel_path} ... 错误: {error}")
                    stats["errors"] += 1
                else:
                    _save_single_json(output_dir, image_id, f"../images/{Path(rel_path).name}", w, h, persons_output)
                    _update_stats(stats, persons_output)
                    all_viz_data.append((abs_path, persons_output))

                    n = len(persons_output)
                    print(f"[{i + 1}/{total}] {rel_path} ... {n}个人" if n else f"[{i + 1}/{total}] {rel_path} ... 未检测到")

                if (i + 1) % 100 == 0:
                    print(f"  [CHECKPOINT] 已处理 {i + 1}/{total}")

        # 多进程模式下单独保存可视化
        if viz_dir:
            print("[INFO] 生成可视化图片...")
            for abs_path, persons_output in all_viz_data:
                try:
                    raw = np.fromfile(abs_path, dtype=np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if img is not None:
                        # 从 persons_output 还原 face_infos 用于可视化
                        face_infos = []
                        for p in persons_output:
                            bx, by, bw, bh = p["bbox"]
                            face_infos.append({
                                "bbox": [bx, by, bw, bh],
                                "gender": p["gender"],
                                "gender_label": GENDER_LABELS[p["gender"]],
                                "race": p["race"],
                                "race_label": RACE_LABELS[p["race"]] if p["race"] < len(RACE_LABELS) else "Unknown",
                                "gender_conf": p.get("gender_conf", 0),
                                "race_conf": p.get("race_conf", 0),
                            })
                        vis_img = draw_results(img, face_infos, [])
                        vis_path = os.path.join(viz_dir, os.path.basename(abs_path))
                        _, ext = os.path.splitext(vis_path)
                        cv2.imencode(ext if ext else ".jpg", vis_img)[1].tofile(vis_path)
                        del vis_img, img, raw
                except Exception:
                    pass

    else:
        # ── 单进程模式 ──
        for i, img_path in enumerate(images):
            rel_path = os.path.relpath(img_path, input_dir)
            image_id = Path(rel_path).stem
            print(f"[{i + 1}/{total}] {rel_path}", end=" ... ", flush=True)

            try:
                raw = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    print("无法读取，跳过")
                    stats["errors"] += 1
                    continue

                h_orig, w_orig = img.shape[:2]

                # 图片缩放
                scale = 1.0
                if max_resize > 0:
                    img, scale = resize_image(img, max_resize)

                # 人体检测
                person_boxes = detect_persons(img)

                # 人脸检测 + 分类
                face_infos = analyze_image(img, detector_backend, conf_threshold,
                                           race_conf_threshold, race_argmax)

                # 还原到原图坐标
                if scale != 1.0:
                    person_boxes = [(int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale), s)
                                   for (x1, y1, x2, y2, s) in person_boxes]
                    face_infos = scale_boxes(face_infos, scale)

                # 匹配人体和人脸
                persons_output = _match_persons_faces(person_boxes, face_infos)

                # 保存单图 JSON
                _save_single_json(output_dir, image_id, f"../images/{Path(rel_path).name}",
                                  w_orig, h_orig, persons_output)

                # 绘制可视化
                if viz_dir and persons_output:
                    try:
                        raw_orig = np.fromfile(img_path, dtype=np.uint8)
                        img_orig = cv2.imdecode(raw_orig, cv2.IMREAD_COLOR)
                        if img_orig is not None:
                            face_infos_vis = []
                            for p in persons_output:
                                bx, by, bw, bh = p["bbox"]
                                face_infos_vis.append({
                                    "bbox": [bx, by, bw, bh],
                                    "gender": p["gender"],
                                    "gender_label": GENDER_LABELS[p["gender"]],
                                    "race": p["race"],
                                    "race_label": RACE_LABELS[p["race"]] if p["race"] < len(RACE_LABELS) else "Unknown",
                                    "gender_conf": p.get("gender_conf", 0),
                                    "race_conf": p.get("race_conf", 0),
                                })
                            vis_img = draw_results(img_orig, face_infos_vis, [])
                            vis_path = os.path.join(viz_dir, os.path.basename(img_path))
                            _, ext = os.path.splitext(vis_path)
                            cv2.imencode(ext if ext else ".jpg", vis_img)[1].tofile(vis_path)
                            del vis_img, img_orig, raw_orig
                    except Exception as ve:
                        print(f"[WARN] 可视化保存失败: {ve}", end=" ")

                del img, raw
                _update_stats(stats, persons_output)

                n = len(persons_output)
                if n:
                    summary = [f"{GENDER_LABELS[p['gender']]}/{RACE_LABELS[p['race']] if p['race'] < len(RACE_LABELS) else '?'}" for p in persons_output]
                    print(f"{n}个人 -> {summary}")
                else:
                    print("未检测到")

            except KeyboardInterrupt:
                print("\n[WARN] 用户中断，保存已处理结果...")
                break
            except Exception as e:
                print(f"处理异常: {e}")
                stats["errors"] += 1
                continue

            if (i + 1) % 50 == 0:
                gc.collect()

    elapsed = time.time() - start_time

    # 保存汇总文件
    summary = {
        "metadata": {
            "tool": "F5 Gender & Race Auto-Labeling Tool v3",
            "model": "deepface",
            "person_detector": "yolov8n",
            "face_detector_backend": detector_backend,
            "conf_threshold": conf_threshold,
            "race_conf_threshold": race_conf_threshold if not race_argmax else "argmax",
            "gpu": gpu_info,
            "total_images": stats["total_images"],
            "total_persons": stats["total_persons"],
            "errors": stats["errors"],
            "elapsed_seconds": round(elapsed, 2),
            "fps": round(stats["total_images"] / elapsed, 2) if elapsed > 0 else 0,
        },
        "statistics": {
            "total_persons": stats["total_persons"],
            "no_person_images": stats["no_person"],
            "female": stats["female"],
            "male_or_unknown": stats["male_or_unknown"],
            "yellow": stats["yellow"],
            "white": stats["white"],
            "brown": stats["brown"],
            "race_unknown": stats["race_unknown"],
            "error_images": stats["errors"],
        },
    }

    summary_path = os.path.join(output_dir, "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    print("=" * 60)
    print(f"[DONE] 标注完成!")
    print(f"  图片总数: {stats['total_images']} (错误: {stats['errors']})")
    print(f"  人体总数: {stats['total_persons']} (无人体: {stats['no_person']})")
    print(f"  性别 - female: {stats['female']}, male_or_unknown: {stats['male_or_unknown']}")
    print(f"  人种 - yellow: {stats['yellow']}, white: {stats['white']}, brown: {stats['brown']}, race_unknown: {stats['race_unknown']}")
    print(f"  耗时: {elapsed:.1f}s ({stats['total_images'] / elapsed:.1f} fps)")
    print(f"  输出目录: {output_dir}")
    print(f"  汇总文件: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="F5 性别与人种自动标注工具 v3 (deepface + 人体检测)")
    parser.add_argument("--input", "-i", required=True, help="输入图片文件夹路径")
    parser.add_argument("--output", "-o", default="./output_labels", help="输出目录路径 (每张图一个JSON)")
    parser.add_argument("--conf", "-c", type=float, default=0.6, help="性别置信度阈值 (默认: 0.6)")
    parser.add_argument("--race-conf", type=float, default=0.3, help="人种置信度阈值 (默认: 0.3, 人种模型置信度天然较低)")
    parser.add_argument("--race-argmax", action="store_true", help="人种分类使用 argmax 模式（直接取最高分，完全消除 Unknown）")
    parser.add_argument("--detector", "-d", default="retinaface",
                        choices=["opencv", "mtcnn", "retinaface", "mediapipe", "ssd", "yolov8n", "fastmtcnn"],
                        help="人脸检测后端 (默认: retinaface)")
    parser.add_argument("--viz-dir", "-v", default=None,
                        help="可视化输出目录 (将保存带人体框+人脸框标注的图片)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="并行进程数 (默认: 1, -1=自动使用所有CPU核心)")
    parser.add_argument("--resize", type=int, default=0,
                        help="图片缩放: 长边最大像素数 (默认: 0=不缩放, 推荐: 1024)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] 输入路径不存在或不是目录: {args.input}")
        sys.exit(1)

    run_labeling(args.input, args.output, args.conf, args.detector,
                 args.viz_dir, args.workers, args.resize,
                 args.race_conf, args.race_argmax)


if __name__ == "__main__":
    main()
