"""
F5 性别与人种自动标注工具 v3 — GUI 界面
提供可视化操作界面，支持标注内容选择、精度选择、路径选择、结果预览。
"""

import json
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# 确保能找到主脚本
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# ─────────── 精度预设 ───────────

PRESETS = {
    "高精度": {"detector": "retinaface", "conf": 0.8, "race_conf": 0.5, "race_argmax": False},
    "均衡":   {"detector": "retinaface", "conf": 0.6, "race_conf": 0.3, "race_argmax": False},
    "快速":   {"detector": "opencv",      "conf": 0.6, "race_conf": 0.3, "race_argmax": False},
    "argmax":  {"detector": "retinaface", "conf": 0.6, "race_conf": 0.3, "race_argmax": True},
}


class LabelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("F5 标注工具 v3")
        self.root.geometry("900x750")
        self.root.resizable(True, True)

        # 状态
        self.is_running = False
        self.images = []
        self.current_idx = 0

        self._build_ui()

    def _build_ui(self):
        # ── 顶部：路径选择 ──
        path_frame = ttk.LabelFrame(self.root, text="路径设置", padding=10)
        path_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(path_frame, text="输入目录:").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.input_var, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="浏览", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(path_frame, text="输出目录:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_var = tk.StringVar(value="./output_labels")
        ttk.Entry(path_frame, textvariable=self.output_var, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="浏览", command=self._browse_output).grid(row=1, column=2)

        ttk.Label(path_frame, text="可视化目录:").grid(row=2, column=0, sticky="w")
        self.viz_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.viz_var, width=60).grid(row=2, column=1, padx=5)
        ttk.Button(path_frame, text="浏览", command=self._browse_viz).grid(row=2, column=2)

        # ── 中部左侧：标注选项 + 精度 ──
        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(fill="x", padx=10, pady=5)

        # 标注内容开关
        opt_frame = ttk.LabelFrame(mid_frame, text="标注内容", padding=10)
        opt_frame.pack(side="left", fill="y", padx=(0, 5))

        self.enable_person = tk.BooleanVar(value=True)
        self.enable_face = tk.BooleanVar(value=True)
        self.enable_gender = tk.BooleanVar(value=True)
        self.enable_race = tk.BooleanVar(value=True)

        ttk.Checkbutton(opt_frame, text="人体检测 (YOLOv8)", variable=self.enable_person).pack(anchor="w")
        ttk.Checkbutton(opt_frame, text="人脸检测 (DeepFace)", variable=self.enable_face).pack(anchor="w")
        ttk.Checkbutton(opt_frame, text="性别分类", variable=self.enable_gender).pack(anchor="w")
        ttk.Checkbutton(opt_frame, text="人种分类", variable=self.enable_race).pack(anchor="w")

        # 精度预设
        preset_frame = ttk.LabelFrame(mid_frame, text="精度预设", padding=10)
        preset_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.preset_var = tk.StringVar(value="均衡")
        for name in PRESETS:
            ttk.Radiobutton(preset_frame, text=name, variable=self.preset_var,
                            value=name, command=self._apply_preset).pack(anchor="w")

        # 参数配置
        param_frame = ttk.LabelFrame(mid_frame, text="参数配置", padding=10)
        param_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

        ttk.Label(param_frame, text="检测后端:").grid(row=0, column=0, sticky="w")
        self.detector_var = tk.StringVar(value="retinaface")
        det_combo = ttk.Combobox(param_frame, textvariable=self.detector_var, width=12,
                                  values=["retinaface", "mtcnn", "opencv", "mediapipe", "ssd", "yolov8n", "fastmtcnn"],
                                  state="readonly")
        det_combo.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="性别阈值:").grid(row=1, column=0, sticky="w")
        self.conf_var = tk.DoubleVar(value=0.6)
        conf_scale = ttk.Scale(param_frame, from_=0.3, to=0.95, variable=self.conf_var, orient="horizontal", length=100)
        conf_scale.grid(row=1, column=1, padx=5, pady=2)
        self.conf_label = ttk.Label(param_frame, text="0.60")
        self.conf_label.grid(row=1, column=2)
        conf_scale.configure(command=lambda v: self.conf_label.configure(text=f"{float(v):.2f}"))

        ttk.Label(param_frame, text="人种阈值:").grid(row=2, column=0, sticky="w")
        self.race_conf_var = tk.DoubleVar(value=0.3)
        race_scale = ttk.Scale(param_frame, from_=0.1, to=0.8, variable=self.race_conf_var, orient="horizontal", length=100)
        race_scale.grid(row=2, column=1, padx=5, pady=2)
        self.race_conf_label = ttk.Label(param_frame, text="0.30")
        self.race_conf_label.grid(row=2, column=2)
        race_scale.configure(command=lambda v: self.race_conf_label.configure(text=f"{float(v):.2f}"))

        self.race_argmax_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="人种 argmax", variable=self.race_argmax_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(param_frame, text="并行进程:").grid(row=4, column=0, sticky="w")
        self.workers_var = tk.IntVar(value=1)
        ttk.Spinbox(param_frame, from_=1, to=16, textvariable=self.workers_var, width=5).grid(row=4, column=1, padx=5, pady=2)

        # 初始化预设
        self._apply_preset()

        # ── 运行按钮 + 进度 ──
        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", padx=10, pady=5)

        self.run_btn = ttk.Button(run_frame, text="▶ 开始标注", command=self._start_labeling)
        self.run_btn.pack(side="left")

        self.stop_btn = ttk.Button(run_frame, text="■ 停止", command=self._stop_labeling, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(run_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.pack(side="left", padx=10)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(run_frame, textvariable=self.status_var).pack(side="left")

        # ── 下部：预览 + 日志 ──
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 预览区域
        preview_frame = ttk.LabelFrame(bottom_frame, text="标注预览", padding=5)
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.preview_canvas = tk.Canvas(preview_frame, bg="#2b2b2b", width=400, height=350)
        self.preview_canvas.pack(fill="both", expand=True)

        # 预览导航
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill="x", pady=5)
        ttk.Button(nav_frame, text="◀ 上一张", command=self._prev_image).pack(side="left")
        ttk.Button(nav_frame, text="下一张 ▶", command=self._next_image).pack(side="left", padx=5)
        self.preview_label = ttk.Label(nav_frame, text="")
        self.preview_label.pack(side="left", padx=10)
        ttk.Button(nav_frame, text="刷新预览", command=self._refresh_preview).pack(side="right")

        # 日志区域
        log_frame = ttk.LabelFrame(bottom_frame, text="运行日志", padding=5)
        log_frame.pack(side="right", fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=15, width=50, state="disabled", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

    # ─────────── 路径选择 ───────────

    def _browse_input(self):
        path = filedialog.askdirectory(title="选择输入图片目录")
        if path:
            self.input_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(path)

    def _browse_viz(self):
        path = filedialog.askdirectory(title="选择可视化输出目录")
        if path:
            self.viz_var.set(path)

    # ─────────── 精度预设 ───────────

    def _apply_preset(self):
        name = self.preset_var.get()
        if name in PRESETS:
            p = PRESETS[name]
            self.detector_var.set(p["detector"])
            self.conf_var.set(p["conf"])
            self.conf_label.configure(text=f"{p['conf']:.2f}")
            self.race_conf_var.set(p["race_conf"])
            self.race_conf_label.configure(text=f"{p['race_conf']:.2f}")
            self.race_argmax_var.set(p["race_argmax"])

    # ─────────── 日志 ───────────

    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ─────────── 标注流程 ───────────

    def _start_labeling(self):
        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()

        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("错误", "请选择有效的输入目录")
            return
        if not output_dir:
            messagebox.showerror("错误", "请设置输出目录")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        viz_dir = self.viz_var.get().strip()
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)

        self.is_running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("正在初始化...")
        self.progress_var.set(0)

        # 在后台线程运行
        thread = threading.Thread(target=self._run_labeling, daemon=True)
        thread.start()

    def _stop_labeling(self):
        self.is_running = False
        self.status_var.set("正在停止...")
        self._log("[INFO] 用户请求停止...")

    def _run_labeling(self):
        try:
            # 延迟导入避免启动慢
            from label_gender_race import (
                collect_images, detect_persons, analyze_image,
                _match_persons_faces, _save_single_json, _update_stats,
                _get_gpu_info, draw_results, resize_image, scale_boxes,
                GENDER_LABELS, RACE_LABELS, _json_default
            )
            import cv2
            import numpy as np

            input_dir = self.input_var.get().strip()
            output_dir = self.output_var.get().strip()
            viz_dir = self.viz_var.get().strip() or None

            detector = self.detector_var.get()
            conf = self.conf_var.get()
            race_conf = self.race_conf_var.get()
            race_argmax = self.race_argmax_var.get()
            workers = self.workers_var.get()

            enable_person = self.enable_person.get()
            enable_face = self.enable_face.get()
            enable_gender = self.enable_gender.get()
            enable_race = self.enable_race.get()

            self.root.after(0, lambda: self._log(f"[INFO] GPU: {_get_gpu_info()}"))
            self.root.after(0, lambda: self._log(f"[INFO] 检测后端: {detector}"))
            self.root.after(0, lambda: self._log(f"[INFO] 性别阈值: {conf:.2f}, 人种阈值: {race_conf:.2f}"))
            self.root.after(0, lambda: self._log(f"[INFO] 标注内容: person={enable_person}, face={enable_face}, gender={enable_gender}, race={enable_race}"))

            skip = {"viz", "labels", "output"}
            images = collect_images(input_dir, skip_dirs=skip)
            total = len(images)

            if total == 0:
                self.root.after(0, lambda: self._log("[ERROR] 未找到图片"))
                self.root.after(0, lambda: self.status_var.set("未找到图片"))
                return

            self.images = images
            self.root.after(0, lambda: self._log(f"[INFO] 共找到 {total} 张图片"))
            self.root.after(0, lambda: self.status_var.set(f"标注中... 0/{total}"))

            stats = {
                "total_images": 0, "total_persons": 0,
                "female": 0, "male_or_unknown": 0,
                "yellow": 0, "white": 0, "brown": 0, "race_unknown": 0,
                "no_person": 0, "errors": 0,
            }
            import time
            start_time = time.time()

            for i, img_path in enumerate(images):
                if not self.is_running:
                    break

                rel_path = os.path.relpath(img_path, input_dir)
                image_id = Path(rel_path).stem

                try:
                    raw = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if img is None:
                        stats["errors"] += 1
                        continue

                    h_orig, w_orig = img.shape[:2]

                    # 人体检测
                    person_boxes = []
                    if enable_person:
                        person_boxes = detect_persons(img)

                    # 人脸检测 + 分类
                    face_infos = []
                    if enable_face:
                        face_infos = analyze_image(img, detector, conf, race_conf, race_argmax)

                        # 如果不需要性别/人种，清除对应字段
                        if not enable_gender:
                            for f in face_infos:
                                f["gender"] = 1
                                f["gender_conf"] = 0
                        if not enable_race:
                            for f in face_infos:
                                f["race"] = 3
                                f["race_conf"] = 0

                    # 匹配
                    persons_output = _match_persons_faces(person_boxes, face_infos)

                    # 保存 JSON
                    _save_single_json(output_dir, image_id, f"../images/{Path(rel_path).name}",
                                      w_orig, h_orig, persons_output)
                    _update_stats(stats, persons_output)

                    # 可视化
                    if viz_dir and persons_output:
                        face_vis = []
                        for p in persons_output:
                            bx, by, bw, bh = p["bbox"]
                            face_vis.append({
                                "bbox": [bx, by, bw, bh],
                                "gender": p.get("gender", 1),
                                "gender_label": GENDER_LABELS[p.get("gender", 1)],
                                "race": p.get("race", 3),
                                "race_label": RACE_LABELS[p.get("race", 3)] if p.get("race", 3) < len(RACE_LABELS) else "Unknown",
                                "gender_conf": p.get("gender_conf", 0),
                                "race_conf": p.get("race_conf", 0),
                            })
                        vis_img = draw_results(img, face_vis, [])
                        vis_path = os.path.join(viz_dir, os.path.basename(img_path))
                        _, ext = os.path.splitext(vis_path)
                        cv2.imencode(ext if ext else ".jpg", vis_img)[1].tofile(vis_path)

                    del img, raw

                    n = len(persons_output)
                    msg = f"[{i+1}/{total}] {rel_path} ... {n}个人"
                    self.root.after(0, lambda m=msg: self._log(m))
                    self.root.after(0, lambda v=(i+1)/total*100: self.progress_var.set(v))
                    self.root.after(0, lambda s=f"标注中... {i+1}/{total}": self.status_var.set(s))

                    # 更新预览
                    if viz_dir and i < 3:
                        self.current_idx = i
                        self.root.after(0, self._refresh_preview)

                except Exception as e:
                    stats["errors"] += 1
                    self.root.after(0, lambda m=f"[ERROR] {rel_path}: {e}": self._log(m))

            elapsed = time.time() - start_time

            # 保存汇总
            summary = {
                "metadata": {
                    "tool": "F5 Gender & Race Auto-Labeling Tool v3 (GUI)",
                    "detector": detector,
                    "conf_threshold": conf,
                    "race_conf_threshold": race_conf if not race_argmax else "argmax",
                    "enable_person": enable_person,
                    "enable_face": enable_face,
                    "enable_gender": enable_gender,
                    "enable_race": enable_race,
                    "total_images": stats["total_images"],
                    "total_persons": stats["total_persons"],
                    "errors": stats["errors"],
                    "elapsed_seconds": round(elapsed, 2),
                },
                "statistics": {
                    "total_persons": stats["total_persons"],
                    "female": stats["female"],
                    "male_or_unknown": stats["male_or_unknown"],
                    "yellow": stats["yellow"],
                    "white": stats["white"],
                    "brown": stats["brown"],
                    "race_unknown": stats["race_unknown"],
                },
            }
            summary_path = os.path.join(output_dir, "_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

            self.root.after(0, lambda: self._log("=" * 50))
            self.root.after(0, lambda: self._log(f"[DONE] 标注完成! {stats['total_images']}张, 错误{stats['errors']}张, 耗时{elapsed:.1f}s"))
            self.root.after(0, lambda: self.status_var.set(f"完成! {stats['total_images']}张"))

        except Exception as e:
            self.root.after(0, lambda: self._log(f"[FATAL] {e}"))
            self.root.after(0, lambda: self.status_var.set("出错"))
        finally:
            self.root.after(0, lambda: self.run_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            self.is_running = False

    # ─────────── 预览 ───────────

    def _refresh_preview(self):
        viz_dir = self.viz_var.get().strip()
        if not viz_dir or not os.path.isdir(viz_dir):
            return

        viz_images = sorted([f for f in os.listdir(viz_dir) if f.lower().endswith(('.jpg', '.png'))])
        if not viz_images:
            return

        if self.current_idx >= len(viz_images):
            self.current_idx = 0

        img_path = os.path.join(viz_dir, viz_images[self.current_idx])
        self._show_image(img_path)
        self.preview_label.configure(text=f"{self.current_idx + 1}/{len(viz_images)} - {viz_images[self.current_idx]}")

    def _prev_image(self):
        viz_dir = self.viz_var.get().strip()
        if not viz_dir:
            return
        viz_images = sorted([f for f in os.listdir(viz_dir) if f.lower().endswith(('.jpg', '.png'))])
        if viz_images:
            self.current_idx = (self.current_idx - 1) % len(viz_images)
            self._refresh_preview()

    def _next_image(self):
        viz_dir = self.viz_var.get().strip()
        if not viz_dir:
            return
        viz_images = sorted([f for f in os.listdir(viz_dir) if f.lower().endswith(('.jpg', '.png'))])
        if viz_images:
            self.current_idx = (self.current_idx + 1) % len(viz_images)
            self._refresh_preview()

    def _show_image(self, img_path):
        try:
            from PIL import Image, ImageTk
            img = Image.open(img_path)
            # 缩放到预览区域大小
            canvas_w = self.preview_canvas.winfo_width() or 400
            canvas_h = self.preview_canvas.winfo_height() or 350
            img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center")
            self.preview_canvas._photo = photo  # 保持引用
        except ImportError:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(200, 175, text="需要安装 Pillow:\npip install Pillow",
                                             fill="white", font=("Arial", 12))
        except Exception as e:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(200, 175, text=f"预览失败: {e}",
                                             fill="white", font=("Arial", 10))


def main():
    root = tk.Tk()
    app = LabelGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
