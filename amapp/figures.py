
import os
import random

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from .config import DIOR_CLASSES, SENSITIVITY_TABLE
from .metrics import compute_residual_detectability
from .models import get_model_dior
from .pipeline_whole_object import run_amapp_linear
from .scoring import compute_context, compute_exposure, compute_mapr, compute_uniqueness

# PART U — MANUSCRIPT FIGURES 1-5
# =============================================================================

FIGURE_OUTPUT_DIR = "/kaggle/working/figures"

def make_figure1_pipeline_diagram(save_path=None):
    save_path = save_path or os.path.join(FIGURE_OUTPUT_DIR, "figure1_pipeline.png")

    stages = [
        "Input\nImage", "Object\nDetection", "Confidence\nCalibration",
        "MAPR\nRisk Score", "Adaptive\nOperator", "Protected\nImage",
        "Re-detection\n(same detector)",
    ]

    fig, ax = plt.subplots(figsize=(16, 3))
    n = len(stages)
    box_w, box_h, gap = 1.8, 1.0, 0.6
    x = 0.0

    for i, label in enumerate(stages):
        rect = Rectangle((x, 0), box_w, box_h, facecolor="#DCE9F1",
                          edgecolor="#2F4858", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, box_h / 2, label, ha="center", va="center",
                fontsize=10, wrap=True)
        if i < n - 1:
            arrow = FancyArrowPatch((x + box_w, box_h / 2),
                                     (x + box_w + gap, box_h / 2),
                                     arrowstyle="-|>", mutation_scale=15,
                                     color="#2F4858", linewidth=1.5)
            ax.add_patch(arrow)
        x += box_w + gap

    ax.set_xlim(-0.3, x)
    ax.set_ylim(-0.3, box_h + 0.3)
    ax.axis("off")
    ax.set_title("Figure 1. AMAPP pipeline", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path

def make_figure2_reliability_diagram(reliability_data, n_bins=10, save_path=None):
    save_path = save_path or os.path.join(FIGURE_OUTPUT_DIR, "figure2_reliability.png")

    raw_conf = reliability_data["test_confidences_raw"]
    cal_conf = reliability_data["test_confidences_calibrated"]
    correct = reliability_data["test_corrects"]

    def bin_stats(conf, correct, n_bins):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers, accs, confs = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            idx = (conf >= lo) & (conf < hi)
            if idx.any():
                bin_centers.append((lo + hi) / 2)
                accs.append(correct[idx].mean())
                confs.append(conf[idx].mean())
        return np.array(bin_centers), np.array(accs), np.array(confs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, conf, title, ece_key in [
        (axes[0], raw_conf, "Before calibration", "ece_before"),
        (axes[1], cal_conf, "After calibration", "ece_after"),
    ]:
        centers, accs, confs = bin_stats(conf, correct, n_bins)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.bar(centers, accs, width=1.0 / n_bins, edgecolor="#2F4858",
               color="#DCE9F1", label="Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy (fraction correct)")
        ax.set_title(f"{title}\nECE = {reliability_data[ece_key]:.4f}")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Figure 2. Confidence reliability, before vs. after temperature scaling "
                  f"(T={reliability_data['T']:.3f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path

def compute_mapr_heatmap(image, detections, img_shape):
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)
    for cls_name, confidence, bbox in detections:
        exposure = compute_exposure(bbox, img_shape)
        uniqueness = compute_uniqueness(cls_name, detections)
        context = compute_context(bbox, detections)
        mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], mapr)
    return heatmap

def make_figure3_sequence(image_path, save_path=None, conf_threshold=0.4):
    save_path = save_path or os.path.join(FIGURE_OUTPUT_DIR, "figure3_sequence.png")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    img_shape = image.shape

    results = get_model_dior().predict(source=image_path, save=False, conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = DIOR_CLASSES[cls_id]
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()
            detections.append((cls_name, confidence, bbox))

    detection_vis = image.copy()
    for cls_name, confidence, bbox in detections:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(detection_vis, f"{cls_name} {confidence:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    heatmap = compute_mapr_heatmap(image, detections, img_shape) if detections else np.zeros(img_shape[:2])

    protected, scored = (run_amapp_linear(image, detections, img_shape)
                         if detections else (image.copy(), []))

    original_count = len(detections)
    if original_count > 0:
        residual = compute_residual_detectability(protected, get_model_dior(), DIOR_CLASSES, original_count)
    else:
        residual = 0.0

    redetect_results = get_model_dior().predict(source=protected, save=False, conf=conf_threshold, verbose=False)
    redetect_vis = protected.copy()
    for r in redetect_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cv2.rectangle(redetect_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    panels = [
        (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "1. Original"),
        (cv2.cvtColor(detection_vis, cv2.COLOR_BGR2RGB), "2. Detection"),
        (heatmap, "3. MAPR risk map"),
        (cv2.cvtColor(protected, cv2.COLOR_BGR2RGB), "4. Protected (linear)"),
        (cv2.cvtColor(redetect_vis, cv2.COLOR_BGR2RGB), f"5. Re-detection\n(same detector)\nresidual={residual:.2f}"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (img, title) in zip(axes, panels):
        if img.ndim == 2:
            im = ax.imshow(img, cmap="hot", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("Figure 3. Representative protection sequence", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path, {"detections": detections, "residual": residual}

def find_multi_risk_image(val_dir, sample_size=100, seed=42, conf_threshold=0.4,
                           min_detections=3, max_detections=8):
    random.seed(seed)
    all_images = sorted(os.listdir(val_dir))
    sample_images = random.sample(all_images, min(sample_size, len(all_images)))

    best_candidate = None  # (num_detections, img_path, detections, maprs)

    for fname in sample_images:
        img_path = os.path.join(val_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue
        img_shape = image.shape

        results = get_model_dior().predict(source=img_path, save=False, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = DIOR_CLASSES[cls_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()
                detections.append((cls_name, confidence, bbox))

        if not (min_detections <= len(detections) <= max_detections):
            continue

        maprs = []
        for cls_name, confidence, bbox in detections:
            exposure = compute_exposure(bbox, img_shape)
            uniqueness = compute_uniqueness(cls_name, detections)
            context = compute_context(bbox, detections)
            maprs.append(compute_mapr(cls_name, confidence, exposure, uniqueness, context))

        spans_risk_range = (max(maprs) - min(maprs)) > 0.25

        has_overlap = False
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                b1, b2 = detections[i][2], detections[j][2]
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                if ix2 > ix1 and iy2 > iy1:
                    has_overlap = True

        if spans_risk_range and has_overlap:
            if best_candidate is None or len(detections) < best_candidate[0]:
                best_candidate = (len(detections), img_path, detections, maprs)

    if best_candidate is None:
        return None, None, None
    _, img_path, detections, maprs = best_candidate
    return img_path, detections, maprs

def make_figure4_multi_risk(image_path, detections, maprs, save_path=None, crop_padding=60):
    save_path = save_path or os.path.join(FIGURE_OUTPUT_DIR, "figure4_multi_risk.png")

    image = cv2.imread(image_path)
    vis = image.copy()

    def risk_tier(mapr):
        if mapr < 0.4:
            return "low", (0, 255, 0)
        elif mapr < 0.7:
            return "medium", (0, 165, 255)
        else:
            return "high", (0, 0, 255)

    for (cls_name, confidence, bbox), mapr in zip(detections, maprs):
        tier, color = risk_tier(mapr)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} ({tier}, {mapr:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 10
        cv2.rectangle(vis, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), (255, 255, 255), -1)
        cv2.putText(vis, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    xs1 = [int(bbox[0]) for _, _, bbox in detections]
    ys1 = [int(bbox[1]) for _, _, bbox in detections]
    xs2 = [int(bbox[2]) for _, _, bbox in detections]
    ys2 = [int(bbox[3]) for _, _, bbox in detections]
    h, w = vis.shape[:2]
    cx1 = max(min(xs1) - crop_padding, 0)
    cy1 = max(min(ys1) - crop_padding, 0)
    cx2 = min(max(xs2) + crop_padding, w)
    cy2 = min(max(ys2) + crop_padding, h)
    vis_cropped = vis[cy1:cy2, cx1:cx2]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(cv2.cvtColor(vis_cropped, cv2.COLOR_BGR2RGB))
    ax.set_title("Figure 4. Multi-attribute scene: low/medium/high risk, "
                 "including an overlapping-region case", fontsize=11)
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(color="green", label="Low risk (MAPR < 0.4)"),
        mpatches.Patch(color="orange", label="Medium risk (0.4-0.7)"),
        mpatches.Patch(color="red", label="High risk (>= 0.7)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path

def make_figure5_tradeoff_curve(save_path=None):
    save_path = save_path or os.path.join(FIGURE_OUTPUT_DIR, "figure5_tradeoff.png")

    methods = {
        "Original":            (-1.5, None),
        "Global Gaussian blur": (75.5, 21.94),
        "Fixed masking":       (93.5, 18.43),
        "Single-attribute":    (43.6, 22.97),
        "AMAPP (linear)":      (74.8, 29.82),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (risk_reduction, psnr) in methods.items():
        if psnr is None:
            continue
        color = "#C0392B" if name.startswith("AMAPP") else "#2F4858"
        marker = "*" if name.startswith("AMAPP") else "o"
        size = 220 if name.startswith("AMAPP") else 100
        ax.scatter(risk_reduction, psnr, s=size, color=color, marker=marker, zorder=3)
        ax.annotate(name, (risk_reduction, psnr), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)

    ax.set_xlabel("Risk reduction (%)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Figure 5. Privacy-utility trade-off (DIOR, whole-object, linear)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path
