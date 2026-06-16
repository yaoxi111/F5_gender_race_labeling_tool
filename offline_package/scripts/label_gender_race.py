"""
F5 性别与人种自动标注工具 v2 — 离线版
人体检测 + 人脸检测 + 性别分类 + 人种分类
所有模型权重已内置，无需联网。

用法:
    python label_gender_race.py -i <图片文件夹> -o <输出JSON> [-c 0.6] [-d retinaface] [-v viz目录]

依赖:
    pip install deepface opencv-python tf-keras ultralytics numpy
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

# ─────────── 路径配置 ───────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_SCRIPT_DIR)  # offline_package 根目录
_MODELS_DIR = os.path.join(_PKG_DIR, "models")

# DeepFace 权重目录
_DEEPFACE_WEIGHTS = os.path.join(_MODELS_DIR, "deepface")
os.environ["DEEPFACE_HOME"] = _MODELS_DIR

# YOLOv8 模型路径
_YOLO_MODEL_PATH = os.path.join(_MODELS_DIR, "yolov8n.pt")

# 编码修复
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ─────────── 常量 ───────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

COLOR_PERSON = (0, 255, 0)
COLOR_MALE = (255, 128, 0)
COLOR_FEMALE = (0, 80, 255)
COLOR_UNKNOWN_G = (180, 0, 180)

GENDER_MAP = {"Woman": 0, "Man": 1}
GENDER_LABELS = ["Female", "Male", "Unknown"]

RACE_MAP = {
    "asian": 0, "white": 1, "middle eastern": 2,
    "indian": 3, "latino": 4, "black": 5,
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


# ─────────── 人体检测 (YOLOv8) ───────────

_yolo_model = None

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        if os.path.exists(_YOLO_MODEL_PATH):
            _yolo_model = YOLO(_YOLO_MODEL_PATH)
        else:
            print(f"[WARN] YOLOv8 模型未找到: {_YOLO_MODEL_PATH}")
            print(f"[WARN] 将跳过人体检测")
            return None
    return _yolo_model


def detect_persons(img, conf_threshold=0.5):
    model = _get_yolo_model()
    if model is None:
        return []
    try:
        results = model(img, conf=conf_threshold, verbose=False, classes=[0])
        persons = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                score = float(box.conf[0])
                persons.append((int(x1), int(y1), int(x2), int(y2), score))
        return persons
    except Exception as e:
        print(f"[WARN] 人体检测异常: {e}")
        return []


# ─────────── 人脸分析 (DeepFace) ───────────

def analyze_image(img, detector_backend, conf_threshold):
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

    face_infos = []
    for r in demographies:
        region = r.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        if w < 30 or h < 30:
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

    return face_infos


# ─────────── 可视化 ───────────

def draw_results(img, face_infos, person_infos):
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 2

    for p in person_infos:
        px1, py1, px2, py2 = p["x1"], p["y1"], p["x2"], p["y2"]
        cv2.rectangle(vis, (px1, py1), (px2, py2), COLOR_PERSON, 3)
        label = "person"
        (tw, tth), _ = cv2.getTextSize(label, font, fs, th)
        ly = py1 - tth - 12 if py1 - tth - 12 > 0 else py1 + 4
        cv2.rectangle(vis, (px1, ly), (px1 + tw + 10, ly + tth + 10), COLOR_PERSON, -1)
        cv2.putText(vis, label, (px1 + 5, ly + tth + 4), font, fs, (255, 255, 255), th)

    for face in face_infos:
        x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
        gid = face["gender"]
        color = COLOR_FEMALE if gid == 0 else (COLOR_MALE if gid == 1 else COLOR_UNKNOWN_G)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        l1 = f"{face['gender_label']}/{face['race_label']}"
        l2 = f"{face['gender_conf']*100:.0f}%/{face['race_conf']*100:.0f}%"
        (tw1, th1), _ = cv2.getTextSize(l1, font, fs, th)
        (tw2, th2), _ = cv2.getTextSize(l2, font, fs, th)
        bw = max(tw1, tw2) + 10
        bh = th1 + th2 + 16
        ly1 = y1 - bh - 4 if y1 - bh - 4 > 0 else y1 + 4
        cv2.rectangle(vis, (x1, ly1), (x1 + bw, ly1 + bh), color, -1)
        cv2.putText(vis, l1, (x1 + 5, ly1 + th1 + 4), font, fs, (255, 255, 255), th)
        cv2.putText(vis, l2, (x1 + 5, ly1 + th1 + th2 + 12), font, fs, (255, 255, 255), th)

    return vis


# ─────────── 主流程（带保护） ───────────

def collect_images(input_dir, skip_dirs=None):
    skip = set(skip_dirs) if skip_dirs else set()
    images = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                images.append(os.path.join(root, f))
    images.sort()
    return images


def _save_checkpoint(output_path, output_data):
    """保存中间结果，防止崩溃丢失"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=_json_default)
    except Exception:
        pass


def run_labeling(input_dir, output_path, conf_threshold, detector_backend, viz_dir=None, checkpoint_interval=50):
    skip = {"viz"} if viz_dir else None
    images = collect_images(input_dir, skip_dirs=skip)
    if not images:
        print(f"[ERROR] 未找到图片: {input_dir}")
        sys.exit(1)

    total = len(images)
    print(f"[INFO] 共找到 {total} 张图片")
    print(f"[INFO] 置信度阈值: {conf_threshold}")
    print(f"[INFO] 检测后端: {detector_backend}")
    print(f"[INFO] 输出文件: {output_path}")
    print(f"[INFO] 检查点间隔: 每 {checkpoint_interval} 张保存一次")
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

    for i, img_path in enumerate(images):
        rel_path = os.path.relpath(img_path, input_dir)
        print(f"[{i + 1}/{total}] {rel_path}", end=" ... ", flush=True)

        try:
            # 读取图片
            raw = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                print("无法读取，跳过")
                stats["errors"] += 1
                continue

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

            # 可视化
            if viz_dir and (face_infos or person_infos):
                try:
                    vis_img = draw_results(img, face_infos, person_infos)
                    vis_path = os.path.join(viz_dir, os.path.basename(img_path))
                    _, ext = os.path.splitext(vis_path)
                    cv2.imencode(ext if ext else ".jpg", vis_img)[1].tofile(vis_path)
                    del vis_img
                except Exception as ve:
                    print(f"[WARN] 可视化保存失败: {ve}", end=" ")

            # 释放图片内存
            del img, raw

            all_labels.append({
                "image_path": rel_path,
                "image_abs_path": os.path.abspath(img_path),
                "person_count": len(person_infos),
                "persons": person_infos,
                "face_count": len(face_infos),
                "faces": face_infos,
            })

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

            parts = []
            if person_infos:
                parts.append(f"{len(person_infos)}个人体")
            if face_infos:
                parts.append(f"{len(face_infos)}张人脸 -> {[f'{f[\"gender_label\"]}/{f[\"race_label\"]}' for f in face_infos]}")
            if not parts:
                parts.append("未检测到")
            print(", ".join(parts))

        except KeyboardInterrupt:
            print("\n[WARN] 用户中断，保存已处理的结果...")
            break
        except Exception as e:
            print(f"处理异常: {e}")
            stats["errors"] += 1
            traceback.print_exc()
            continue

        # 定期保存检查点
        if (i + 1) % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            checkpoint = {
                "metadata": {
                    "tool": "F5 Gender & Race Auto-Labeling Tool v2 (offline)",
                    "status": "checkpoint",
                    "processed": i + 1,
                    "total": total,
                    "elapsed_seconds": round(elapsed, 2),
                },
                "statistics": stats,
                "labels": all_labels,
            }
            _save_checkpoint(output_path, checkpoint)
            print(f"  [CHECKPOINT] 已保存 ({i + 1}/{total})")
            gc.collect()

    elapsed = time.time() - start_time

    output = {
        "metadata": {
            "tool": "F5 Gender & Race Auto-Labeling Tool v2 (offline)",
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

    _save_checkpoint(output_path, output)

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
    parser = argparse.ArgumentParser(description="F5 标注工具 v2 离线版")
    parser.add_argument("--input", "-i", required=True, help="输入图片文件夹")
    parser.add_argument("--output", "-o", default="./output/labels.json", help="输出 JSON 路径")
    parser.add_argument("--conf", "-c", type=float, default=0.6, help="置信度阈值")
    parser.add_argument("--detector", "-d", default="opencv",
                        choices=["opencv", "mtcnn", "retinaface", "mediapipe", "ssd", "yolov8n", "fastmtcnn"],
                        help="人脸检测后端")
    parser.add_argument("--viz-dir", "-v", default=None, help="可视化输出目录")
    parser.add_argument("--checkpoint", type=int, default=50, help="检查点间隔（每 N 张保存一次）")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] 输入路径不存在: {args.input}")
        sys.exit(1)

    print(f"[INFO] 模型目录: {_MODELS_DIR}")
    print(f"[INFO] DeepFace 权重: {_DEEPFACE_WEIGHTS}")
    print(f"[INFO] YOLOv8 模型: {_YOLO_MODEL_PATH}")
    for f in ["gender_model_weights.h5", "race_model_single_batch.h5", "retinaface.h5"]:
        p = os.path.join(_DEEPFACE_WEIGHTS, f)
        print(f"  {'[OK]' if os.path.exists(p) else '[MISSING]'} {f}")
    print(f"  {'[OK]' if os.path.exists(_YOLO_MODEL_PATH) else '[MISSING]'} yolov8n.pt")
    print("-" * 60)

    run_labeling(args.input, args.output, args.conf, args.detector, args.viz_dir, args.checkpoint)


if __name__ == "__main__":
    main()
