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

# 绘图颜色 (BGR)
COLOR_MALE = (255, 128, 0)      # 蓝色 - 男性
COLOR_FEMALE = (0, 80, 255)     # 红色 - 女性
COLOR_UNKNOWN_G = (180, 0, 180) # 紫色 - 未知性别
COLOR_BG = (0, 0, 0)            # 黑色背景

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


# ─────────── 人脸检测 ───────────

# OpenCV DNN 人脸检测器（懒加载）
_dnn_net = None

def _get_dnn_detector():
    global _dnn_net
    if _dnn_net is None:
        # 使用 OpenCV 内置的 YuNet 模型（OpenCV 4.5.4+ 自带）
        # 如果不可用则回退到 Haar 级联
        _dnn_net = "yunet"
    return _dnn_net


def detect_faces(img):
    """
    检测人脸，返回 [(x1, y1, x2, y2), ...] 列表。
    优先使用 OpenCV 内置 FaceDetectorYN，不可用时回退到 Haar 级联。
    """
    h_img, w_img = img.shape[:2]

    # 方式1: OpenCV FaceDetectorYN (OpenCV 4.5.4+)
    try:
        detector = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx",
            "",
            (320, 320),
        )
        detector.setInputSize((w_img, h_img))
        _, faces = detector.detect(img)
        if faces is not None and len(faces) > 0:
            result = []
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                result.append((x, y, x + w, y + h))
            return result
    except Exception:
        pass

    # 方式2: Haar 级联 (处理中文路径: 复制到脚本目录)
    try:
        cascade_src = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade_dst = os.path.join(_SCRIPT_DIR, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_dst):
            import shutil
            shutil.copy2(cascade_src, cascade_dst)
        detector = cv2.CascadeClassifier(cascade_dst)
        if detector.empty():
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]
    except Exception:
        return []


# ─────────── 核心分析 ───────────

def analyze_image(img, detector_backend, conf_threshold):
    """
    对单张图片做人脸检测 + 性别分类 + 人种分类。
    先用 Haar 级联做人脸框定位，再用 DeepFace 对裁剪区域做分类。
    返回 FaceInfo 列表。
    """
    from deepface import DeepFace

    # 第一步: Haar 级联检测人脸框
    face_boxes = detect_faces(img)
    if not face_boxes:
        return []

    h_img, w_img = img.shape[:2]
    face_infos = []

    for (x1, y1, x2, y2) in face_boxes:
        # 裁剪人脸区域，加一点边距
        pad = int(max(x2 - x1, y2 - y1) * 0.1)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w_img, x2 + pad)
        cy2 = min(h_img, y2 + pad)
        face_crop = img[cy1:cy2, cx1:cx2]

        if face_crop.size == 0:
            continue

        # 第二步: DeepFace 分类
        try:
            demographies = DeepFace.analyze(
                img_path=face_crop,
                actions=["gender", "race"],
                detector_backend=detector_backend,
                enforce_detection=False,
                silent=True,
            )
        except Exception:
            # 分类失败时保留框，标记 Unknown
            face_infos.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_score": 0.0,
                "gender": 2, "gender_label": "Unknown", "gender_conf": 0.0,
                "raw_gender_scores": {},
                "race": 6, "race_label": "Unknown", "race_conf": 0.0,
                "raw_race_scores": {},
            })
            continue

        if not demographies:
            continue

        r = demographies[0] if isinstance(demographies, list) else demographies

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
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
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


# ─────────── 可视化绘制 ───────────

def draw_faces(img, face_infos):
    """
    在图片上绘制人脸框和标注标签。
    返回绘制后的图片副本。
    """
    vis = img.copy()
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

        # 计算文字大小，绘制背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        box_w = max(tw1, tw2) + 10
        box_h = th1 + th2 + 16

        # 标签位置: 框上方，若超出图片则放框内
        label_y1 = y1 - box_h - 4
        if label_y1 < 0:
            label_y1 = y1 + 4
        label_y2 = label_y1 + box_h

        # 画标签背景
        cv2.rectangle(vis, (x1, label_y1), (x1 + box_w, label_y2), color, -1)

        # 画文字 (白色)
        cv2.putText(vis, line1, (x1 + 5, label_y1 + th1 + 4), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(vis, line2, (x1 + 5, label_y1 + th1 + th2 + 12), font, font_scale, (255, 255, 255), thickness)

    return vis


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


def run_labeling(input_dir, output_path, conf_threshold, detector_backend, viz_dir=None):
    skip = {"viz"} if viz_dir else None
    images = collect_images(input_dir, skip_dirs=skip)
    if not images:
        print(f"[ERROR] 未找到图片: {input_dir}")
        sys.exit(1)

    print(f"[INFO] 共找到 {len(images)} 张图片")
    print(f"[INFO] 置信度阈值: {conf_threshold}")
    print(f"[INFO] 检测后端: {detector_backend}")
    print(f"[INFO] 输出文件: {output_path}")
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[INFO] 可视化输出: {viz_dir}")
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

        # 绘制可视化并保存
        if viz_dir and face_infos:
            vis_img = draw_faces(img, face_infos)
            vis_path = os.path.join(viz_dir, os.path.basename(img_path))
            _, ext = os.path.splitext(vis_path)
            cv2.imencode(ext if ext else ".jpg", vis_img)[1].tofile(vis_path)

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
    parser.add_argument("--viz-dir", "-v", default=None,
                        help="可视化输出目录 (将保存带人脸框标注的图片)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] 输入路径不存在或不是目录: {args.input}")
        sys.exit(1)

    run_labeling(args.input, args.output, args.conf, args.detector, args.viz_dir)


if __name__ == "__main__":
    main()
