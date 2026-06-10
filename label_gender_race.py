"""
F5 性别与人种自动标注工具 (deepface 版)
人脸检测 + 性别分类 + 人种分类: deepface
输出与 F5 ODOT FaceInfo 结构对齐的 JSON 标注文件。

用法:
    python label_gender_race.py --input <图片文件夹> --output <输出JSON> [--conf <置信度阈值>] [--detector <检测后端>]

依赖:
    pip install deepface opencv-python tf-keras
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 指向本地 weights 目录，避免每次从 GitHub 下载
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DEEPFACE_HOME"] = _SCRIPT_DIR

# 修复 Windows GBK 编码
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ─────────── 常量 ───────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

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


# ─────────── 核心分析 ───────────

def analyze_image(img, detector_backend, conf_threshold):
    """
    对单张图片做人脸检测 + 性别分类 + 人种分类。
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

    face_infos = []
    for r in demographies:
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

        # ── 人脸框 ──
        region = r.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

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

    return face_infos


# ─────────── 主流程 ───────────

def collect_images(input_dir):
    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                images.append(os.path.join(root, f))
    images.sort()
    return images


def run_labeling(input_dir, output_path, conf_threshold, detector_backend):
    images = collect_images(input_dir)
    if not images:
        print(f"[ERROR] 未找到图片: {input_dir}")
        sys.exit(1)

    print(f"[INFO] 共找到 {len(images)} 张图片")
    print(f"[INFO] 置信度阈值: {conf_threshold}")
    print(f"[INFO] 检测后端: {detector_backend}")
    print(f"[INFO] 输出文件: {output_path}")
    print("-" * 60)

    all_labels = []
    stats = {
        "total_images": 0, "total_faces": 0,
        "male": 0, "female": 0, "unknown_gender": 0,
        "asian": 0, "white": 0, "middle_eastern": 0,
        "indian": 0, "latino": 0, "black": 0, "unknown_race": 0,
        "no_face": 0,
    }
    start_time = time.time()

    for i, img_path in enumerate(images):
        rel_path = os.path.relpath(img_path, input_dir)
        print(f"[{i + 1}/{len(images)}] {rel_path}", end=" ... ")

        raw = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            print("无法读取")
            continue

        face_infos = analyze_image(img, detector_backend, conf_threshold)

        all_labels.append({
            "image_path": rel_path,
            "image_abs_path": os.path.abspath(img_path),
            "face_count": len(face_infos),
            "faces": face_infos,
        })

        stats["total_images"] += 1
        stats["total_faces"] += len(face_infos)
        if not face_infos:
            stats["no_face"] += 1
            print("无人脸")
        else:
            for f in face_infos:
                # 性别统计
                g = f["gender"]
                if g == 0:
                    stats["female"] += 1
                elif g == 1:
                    stats["male"] += 1
                else:
                    stats["unknown_gender"] += 1
                # 人种统计
                race_id = f["race"]
                if race_id == 0:
                    stats["asian"] += 1
                elif race_id == 1:
                    stats["white"] += 1
                elif race_id == 2:
                    stats["middle_eastern"] += 1
                elif race_id == 3:
                    stats["indian"] += 1
                elif race_id == 4:
                    stats["latino"] += 1
                elif race_id == 5:
                    stats["black"] += 1
                else:
                    stats["unknown_race"] += 1

            summary = [f"{f['gender_label']}/{f['race_label']}" for f in face_infos]
            print(f"{len(face_infos)}张人脸 -> {summary}")

    elapsed = time.time() - start_time

    output = {
        "metadata": {
            "tool": "F5 Gender & Race Auto-Labeling Tool",
            "model": "deepface",
            "detector_backend": detector_backend,
            "conf_threshold": conf_threshold,
            "total_images": stats["total_images"],
            "total_faces": stats["total_faces"],
            "elapsed_seconds": round(elapsed, 2),
            "fps": round(stats["total_images"] / elapsed, 2) if elapsed > 0 else 0,
        },
        "statistics": {
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
        },
        "labels": all_labels,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=_json_default)

    print("=" * 60)
    print(f"[DONE] 标注完成!")
    print(f"  图片总数: {stats['total_images']}")
    print(f"  人脸总数: {stats['total_faces']}")
    print(f"  性别 - 男性: {stats['male']}, 女性: {stats['female']}, 未知: {stats['unknown_gender']}")
    print(f"  人种 - 亚洲人: {stats['asian']}, 白人: {stats['white']}, "
          f"中东人: {stats['middle_eastern']}, 印度人: {stats['indian']}, "
          f"拉丁裔: {stats['latino']}, 黑人: {stats['black']}, 未知: {stats['unknown_race']}")
    print(f"  无人脸图片: {stats['no_face']}")
    print(f"  耗时: {elapsed:.1f}s ({stats['total_images'] / elapsed:.1f} fps)")
    print(f"  输出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="F5 性别与人种自动标注工具 (deepface)")
    parser.add_argument("--input", "-i", required=True, help="输入图片文件夹路径")
    parser.add_argument("--output", "-o", default="./output/gender_race_labels.json", help="输出 JSON 文件路径")
    parser.add_argument("--conf", "-c", type=float, default=0.6, help="置信度阈值 (默认: 0.6)")
    parser.add_argument("--detector", "-d", default="opencv",
                        choices=["opencv", "mtcnn", "retinaface", "mediapipe", "ssd", "yolov8n", "fastmtcnn"],
                        help="人脸检测后端 (默认: opencv)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] 输入路径不存在或不是目录: {args.input}")
        sys.exit(1)

    run_labeling(args.input, args.output, args.conf, args.detector)


if __name__ == "__main__":
    main()
