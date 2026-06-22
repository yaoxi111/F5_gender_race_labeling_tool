"""
F5 性别与人种自动标注工具 v2 (deepface 版)
人体检测 + 人脸检测 + 性别分类 + 人种分类: deepface + YOLOv8
输出与 F5 ODOT FaceInfo 结构对齐的 JSON 标注文件。

用法:
    python label_gender_race.py -i <图片文件夹> -o <输出JSON> [-c 0.6] [-d retinaface] [-w 4] [--resize 1024]

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

# ─────────── 常量 ───────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# 绘图颜色 (BGR)
COLOR_PERSON = (0, 255, 0)       # 绿色 - 人体
COLOR_MALE = (255, 128, 0)       # 蓝色 - 男性
COLOR_FEMALE = (0, 80, 255)      # 红色 - 女性
COLOR_UNKNOWN_G = (180, 0, 180)  # 紫色 - 未知性别

# 性别映射: deepface 返回 "Woman"/"Man" -> F5 ODOT: 0=Female, 1=Male, 2=Unknown
GENDER_MAP = {"Woman": 0, "Man": 1}
GENDER_LABELS = ["Female", "Male", "Unknown"]

# 人种映射: deepface 返回 6 类 -> 整数 ID
RACE_MAP = {
    "asian": 0,
    "white": 1,
    "middle eastern": 2,
    "indian": 3,
    "latino": 4,
    "black": 5,
}
RACE_LABELS = ["Asian", "White", "Middle Eastern", "Indian", "Latino", "Black", "Unknown"]


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

def analyze_image(img, detector_backend, conf_threshold):
    """
    对单张图片做人脸检测 + 性别分类 + 人种分类。
    使用 DeepFace.analyze() 一步完成检测 + 分类，确保分类精度。
    返回 FaceInfo 列表。
    """
    from deepface import DeepFace

    try:
        demographies = DeepFace.analyze(
            img_path=img,
            actions=["gender", "race"],
            detector_backend=detector_backend,
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return []

    if not demographies:
        return []

    h_img, w_img = img.shape[:2]
    face_infos = []
    for r in demographies:
        # ── 人脸框 ──
        region = r.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        # 过滤掉过小的人脸（宽或高 < 30 像素）
        if w < 30 or h < 30:
            continue

        # 过滤掉过大的"人脸"（覆盖图片 70% 以上 = 检测失败，返回了整张图）
        if w > w_img * 0.7 and h > h_img * 0.7:
            continue
            continue

        # ── 性别 ──
        gender_label = r.get("dominant_gender", "Unknown")
        gender_scores = r.get("gender", {})
        raw_gender_conf = max(gender_scores.values()) if gender_scores else 0.0
        gender_conf = float(raw_gender_conf) / 100.0

        if gender_conf < conf_threshold:
            gender_id = 2  # Unknown
        else:
            gender_id = GENDER_MAP.get(gender_label, 2)

        # ── 人种 ──
        race_label = r.get("dominant_race", "Unknown")
        race_scores = r.get("race", {})
        raw_race_conf = max(race_scores.values()) if race_scores else 0.0
        race_conf = float(raw_race_conf) / 100.0

        if race_conf < conf_threshold:
            race_id = 6  # Unknown
        else:
            race_id = RACE_MAP.get(race_label.lower(), 6)

        face_infos.append({
            "x1": x, "y1": y, "x2": x + w, "y2": y + h,
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
        })

    # 如果 opencv 没检测到有效人脸，用 mtcnn 重试
    if not face_infos and detector_backend == "opencv":
        try:
            demographies = DeepFace.analyze(
                img_path=img,
                actions=["gender", "race"],
                detector_backend="mtcnn",
                enforce_detection=False,
                silent=True,
            )
            if demographies:
                for r in demographies:
                    region = r.get("region", {})
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                    if w < 30 or h < 30:
                        continue
                    if w > w_img * 0.7 and h > h_img * 0.7:
                        continue
                    gender_label = r.get("dominant_gender", "Unknown")
                    gender_scores = r.get("gender", {})
                    raw_gender_conf = max(gender_scores.values()) if gender_scores else 0.0
                    gender_conf = float(raw_gender_conf) / 100.0
                    gender_id = 2 if gender_conf < conf_threshold else GENDER_MAP.get(gender_label, 2)
                    race_label = r.get("dominant_race", "Unknown")
                    race_scores = r.get("race", {})
                    raw_race_conf = max(race_scores.values()) if race_scores else 0.0
                    race_conf = float(raw_race_conf) / 100.0
                    race_id = 6 if race_conf < conf_threshold else RACE_MAP.get(race_label.lower(), 6)
                    face_infos.append({
                        "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                        "det_score": round(gender_conf, 4),
                        "gender": gender_id, "gender_label": GENDER_LABELS[gender_id],
                        "gender_conf": round(gender_conf, 4),
                        "raw_gender_scores": {k: round(float(v) / 100.0, 4) for k, v in gender_scores.items()},
                        "race": race_id, "race_label": RACE_LABELS[race_id],
                        "race_conf": round(race_conf, 4),
                        "raw_race_scores": {k: round(float(v) / 100.0, 4) for k, v in race_scores.items()},
                    })
        except Exception:
            pass

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
        x1, y1 = face["x1"], face["y1"]
        x2, y2 = face["x2"], face["y2"]

        # 性别决定框颜色
        gid = face["gender"]
        if gid == 0:
            color = COLOR_FEMALE
        elif gid == 1:
            color = COLOR_MALE
        else:
            color = COLOR_UNKNOWN_G

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
    """将框坐标按比例还原到原图尺寸。"""
    if scale == 1.0:
        return boxes
    result = []
    for box in boxes:
        if isinstance(box, dict):
            b = dict(box)
            b["x1"] = int(b["x1"] / scale)
            b["y1"] = int(b["y1"] / scale)
            b["x2"] = int(b["x2"] / scale)
            b["y2"] = int(b["y2"] / scale)
            result.append(b)
        elif isinstance(box, (list, tuple)):
            result.append(tuple(int(v / scale) for v in box))
        else:
            result.append(box)
    return result


# ─────────── 单图处理（可被多进程调用）────────────

def _process_single_image(args):
    """
    处理单张图片，返回 (rel_path, abs_path, person_infos, face_infos, error_msg)。
    用于多进程模式。
    """
    img_path, input_dir, detector_backend, conf_threshold, max_resize = args
    rel_path = os.path.relpath(img_path, input_dir)

    try:
        raw = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return (rel_path, os.path.abspath(img_path), [], [], "无法读取")

        # 图片缩放
        scale = 1.0
        if max_resize and max_resize > 0:
            img, scale = resize_image(img, max_resize)

        # 人体检测
        persons = detect_persons(img)
        person_infos = []
        for (px1, py1, px2, py2, pscore) in persons:
            person_infos.append({
                "x1": px1, "y1": py1, "x2": px2, "y2": py2,
                "label": "person", "det_score": round(pscore, 4),
            })

        # 人脸检测 + 分类
        face_infos = analyze_image(img, detector_backend, conf_threshold)

        # 还原到原图坐标
        if scale != 1.0:
            person_infos = scale_boxes(person_infos, scale)
            face_infos = scale_boxes(face_infos, scale)

        del img, raw
        return (rel_path, os.path.abspath(img_path), person_infos, face_infos, None)

    except Exception as e:
        return (rel_path, os.path.abspath(img_path), [], [], str(e))


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


def _update_stats(stats, person_infos, face_infos):
    """更新统计信息。"""
    stats["total_images"] += 1
    stats["total_persons"] += len(person_infos)
    stats["total_faces"] += len(face_infos)
    if not person_infos:
        stats["no_person"] += 1
    if not face_infos:
        stats["no_face"] += 1
    for f in face_infos:
        g = f["gender"]
        if g == 0: stats["female"] += 1
        elif g == 1: stats["male"] += 1
        else: stats["unknown_gender"] += 1
        rid = f["race"]
        if rid == 0: stats["asian"] += 1
        elif rid == 1: stats["white"] += 1
        elif rid == 2: stats["middle_eastern"] += 1
        elif rid == 3: stats["indian"] += 1
        elif rid == 4: stats["latino"] += 1
        elif rid == 5: stats["black"] += 1
        else: stats["unknown_race"] += 1


def run_labeling(input_dir, output_path, conf_threshold, detector_backend,
                 viz_dir=None, workers=1, max_resize=0):
    skip = {"viz"}
    images = collect_images(input_dir, skip_dirs=skip)
    if not images:
        print(f"[ERROR] 未找到图片: {input_dir}")
        sys.exit(1)

    # 自动检测 CPU 核心数
    if workers == -1:
        workers = os.cpu_count() or 4

    print(f"[INFO] 共找到 {len(images)} 张图片")
    print(f"[INFO] 置信度阈值: {conf_threshold}")
    print(f"[INFO] 检测后端: {detector_backend}")
    print(f"[INFO] 并行进程数: {workers}")
    if max_resize > 0:
        print(f"[INFO] 图片缩放: 长边不超过 {max_resize}px")
    print(f"[INFO] 输出文件: {output_path}")
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[INFO] 可视化输出: {viz_dir}")
    print("-" * 60)

    all_labels = []
    stats = {
        "total_images": 0, "total_faces": 0, "total_persons": 0,
        "male": 0, "female": 0, "unknown_gender": 0,
        "asian": 0, "white": 0, "middle_eastern": 0,
        "indian": 0, "latino": 0, "black": 0, "unknown_race": 0,
        "no_face": 0, "no_person": 0, "errors": 0,
    }
    start_time = time.time()
    total = len(images)

    if workers > 1:
        # ── 多进程模式 ──
        import multiprocessing as mp

        task_args = [
            (img_path, input_dir, detector_backend, conf_threshold, max_resize)
            for img_path in images
        ]

        print(f"[INFO] 启动 {workers} 个进程并行处理...")
        with mp.Pool(processes=workers) as pool:
            for i, (rel_path, abs_path, person_infos, face_infos, error) in enumerate(
                pool.imap_unordered(_process_single_image, task_args)
            ):
                if error:
                    print(f"[{i + 1}/{total}] {rel_path} ... 错误: {error}")
                    stats["errors"] += 1
                else:
                    all_labels.append({
                        "image_path": rel_path,
                        "image_abs_path": abs_path,
                        "person_count": len(person_infos),
                        "persons": person_infos,
                        "face_count": len(face_infos),
                        "faces": face_infos,
                    })
                    _update_stats(stats, person_infos, face_infos)

                    parts = []
                    if person_infos:
                        parts.append(f"{len(person_infos)}个人体")
                    if face_infos:
                        parts.append(f"{len(face_infos)}张人脸")
                    if not parts:
                        parts.append("未检测到")
                    print(f"[{i + 1}/{total}] {rel_path} ... {', '.join(parts)}")

                # 定期保存检查点
                if (i + 1) % 100 == 0:
                    print(f"  [CHECKPOINT] 已处理 {i + 1}/{total}")

        # 多进程模式下单独保存可视化
        if viz_dir:
            print("[INFO] 生成可视化图片...")
            for label in all_labels:
                try:
                    img_path = label["image_abs_path"]
                    raw = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if img is not None:
                        vis_img = draw_results(img, label["faces"], label["persons"])
                        vis_path = os.path.join(viz_dir, os.path.basename(img_path))
                        _, ext = os.path.splitext(vis_path)
                        cv2.imencode(ext if ext else ".jpg", vis_img)[1].tofile(vis_path)
                        del vis_img, img, raw
                except Exception:
                    pass

    else:
        # ── 单进程模式（原始流程）──
        for i, img_path in enumerate(images):
            rel_path = os.path.relpath(img_path, input_dir)
            print(f"[{i + 1}/{total}] {rel_path}", end=" ... ", flush=True)

            try:
                raw = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    print("无法读取，跳过")
                    stats["errors"] += 1
                    continue

                # 图片缩放
                scale = 1.0
                if max_resize > 0:
                    img, scale = resize_image(img, max_resize)

                # 人体检测
                persons = detect_persons(img)
                person_infos = []
                for (px1, py1, px2, py2, pscore) in persons:
                    person_infos.append({
                        "x1": px1, "y1": py1, "x2": px2, "y2": py2,
                        "label": "person", "det_score": round(pscore, 4),
                    })

                # 人脸检测 + 分类
                face_infos = analyze_image(img, detector_backend, conf_threshold)

                # 还原到原图坐标
                if scale != 1.0:
                    person_infos = scale_boxes(person_infos, scale)
                    face_infos = scale_boxes(face_infos, scale)

                # 绘制可视化（在原图上绘制）
                if viz_dir and (face_infos or person_infos):
                    try:
                        # 重新读取原图用于可视化
                        raw_orig = np.fromfile(img_path, dtype=np.uint8)
                        img_orig = cv2.imdecode(raw_orig, cv2.IMREAD_COLOR)
                        if img_orig is not None:
                            vis_img = draw_results(img_orig, face_infos, person_infos)
                            vis_path = os.path.join(viz_dir, os.path.basename(img_path))
                            _, ext = os.path.splitext(vis_path)
                            cv2.imencode(ext if ext else ".jpg", vis_img)[1].tofile(vis_path)
                            del vis_img, img_orig, raw_orig
                    except Exception as ve:
                        print(f"[WARN] 可视化保存失败: {ve}", end=" ")

                del img, raw

                all_labels.append({
                    "image_path": rel_path,
                    "image_abs_path": os.path.abspath(img_path),
                    "person_count": len(person_infos),
                    "persons": person_infos,
                    "face_count": len(face_infos),
                    "faces": face_infos,
                })

                _update_stats(stats, person_infos, face_infos)

                parts = []
                if person_infos:
                    parts.append(f"{len(person_infos)}个人体")
                if face_infos:
                    face_summary = [f"{f['gender_label']}/{f['race_label']}" for f in face_infos]
                    parts.append(f"{len(face_infos)}张人脸 -> {face_summary}")
                if not parts:
                    parts.append("未检测到")
                print(", ".join(parts))

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

    output = {
        "metadata": {
            "tool": "F5 Gender & Race Auto-Labeling Tool v2",
            "model": "deepface",
            "person_detector": "yolov8n",
            "face_detector_backend": detector_backend,
            "conf_threshold": conf_threshold,
            "total_images": stats["total_images"],
            "total_persons": stats["total_persons"],
            "total_faces": stats["total_faces"],
            "errors": stats["errors"],
            "elapsed_seconds": round(elapsed, 2),
            "fps": round(stats["total_images"] / elapsed, 2) if elapsed > 0 else 0,
        },
        "statistics": {
            "persons": stats["total_persons"],
            "no_person_images": stats["no_person"],
            "male": stats["male"],
            "female": stats["female"],
            "unknown_gender": stats["unknown_gender"],
            "asian": stats["asian"],
            "white": stats["white"],
            "middle_eastern": stats["middle_eastern"],
            "indian": stats["indian"],
            "latino": stats["latino"],
            "black": stats["black"],
            "unknown_race": stats["unknown_race"],
            "no_face_images": stats["no_face"],
            "error_images": stats["errors"],
        },
        "labels": all_labels,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=_json_default)

    print("=" * 60)
    print(f"[DONE] 标注完成!")
    print(f"  图片总数: {stats['total_images']} (错误: {stats['errors']})")
    print(f"  人体总数: {stats['total_persons']} (无人体: {stats['no_person']})")
    print(f"  人脸总数: {stats['total_faces']} (无人脸: {stats['no_face']})")
    print(f"  性别 - Male: {stats['male']}, Female: {stats['female']}, Unknown: {stats['unknown_gender']}")
    print(f"  人种 - Asian: {stats['asian']}, White: {stats['white']}, Black: {stats['black']}, Unknown: {stats['unknown_race']}")
    print(f"  耗时: {elapsed:.1f}s ({stats['total_images'] / elapsed:.1f} fps)")
    print(f"  输出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="F5 性别与人种自动标注工具 v2 (deepface + 人体检测)")
    parser.add_argument("--input", "-i", required=True, help="输入图片文件夹路径")
    parser.add_argument("--output", "-o", default="./output/gender_race_labels.json", help="输出 JSON 文件路径")
    parser.add_argument("--conf", "-c", type=float, default=0.6, help="置信度阈值 (默认: 0.6)")
    parser.add_argument("--detector", "-d", default="opencv",
                        choices=["opencv", "mtcnn", "retinaface", "mediapipe", "ssd", "yolov8n", "fastmtcnn"],
                        help="人脸检测后端 (默认: opencv)")
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
                 args.viz_dir, args.workers, args.resize)


if __name__ == "__main__":
    main()
