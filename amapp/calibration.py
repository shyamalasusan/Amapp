
import os
import random

import numpy as np

from .config import SENSITIVITY_TABLE
from .models import get_model_dior

# PART Q — CONFIDENCE CALIBRATION (temperature scaling, ECE/D-ECE)
# =============================================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = [float(v) for v in parts[1:]]
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append((cls_id, [x1, y1, x2, y2]))
    return boxes

def collect_confidence_correctness(val_dir, label_dir, sample_size=293, seed=42,
                                    conf_threshold=0.25, iou_threshold=0.5):
    random.seed(seed)
    all_images = sorted(os.listdir(val_dir))
    sample_images = random.sample(all_images, min(sample_size, len(all_images)))

    confidences, corrects, matched_ious = [], [], []

    for fname in sample_images:
        img_path = os.path.join(val_dir, fname)
        img_id = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, img_id + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        gt_boxes = load_yolo_labels(label_path, img_w, img_h)
        gt_claimed = [False] * len(gt_boxes)

        results = get_model_dior().predict(source=img_path, save=False,
                                      conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                detections.append((cls_id, confidence, xyxy))

        detections.sort(key=lambda d: d[1], reverse=True)

        for cls_id, confidence, pred_box in detections:
            best_iou, best_gt_idx = 0.0, -1
            for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes):
                if gt_claimed[gt_idx] or gt_cls != cls_id:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx

            is_correct = best_iou >= iou_threshold
            if is_correct:
                gt_claimed[best_gt_idx] = True

            confidences.append(confidence)
            corrects.append(1 if is_correct else 0)
            matched_ious.append(best_iou if is_correct else 0.0)

    return np.array(confidences), np.array(corrects), np.array(matched_ious)

def compute_ece(confidences, corrects, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (confidences >= lo) & (confidences < hi)
        if idx.any():
            acc = corrects[idx].mean()
            c = confidences[idx].mean()
            ece += idx.mean() * abs(acc - c)
    return ece

def compute_d_ece(confidences, corrects, ious=None, n_bins=10):
    if ious is None:
        return compute_ece(confidences, corrects, n_bins)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    d_ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (confidences >= lo) & (confidences < hi)
        if idx.any():
            quality = np.where(corrects[idx] == 1, ious[idx], 0.0).mean()
            c = confidences[idx].mean()
            d_ece += idx.mean() * abs(quality - c)
    return d_ece

def fit_temperature(confidences, corrects, t_grid=None):
    eps = 1e-6
    conf_clipped = np.clip(confidences, eps, 1 - eps)
    logits = np.log(conf_clipped / (1 - conf_clipped))

    if t_grid is None:
        t_grid = np.linspace(0.05, 3.0, 100)

    best_t, best_nll = 1.0, np.inf
    for t in t_grid:
        calibrated = 1 / (1 + np.exp(-logits / t))
        calibrated = np.clip(calibrated, eps, 1 - eps)
        nll = -np.mean(corrects * np.log(calibrated) + (1 - corrects) * np.log(1 - calibrated))
        if nll < best_nll:
            best_nll, best_t = nll, t

    return best_t

def fit_temperature_d_ece(confidences, corrects, ious, t_grid=None, n_bins=10):
    if t_grid is None:
        t_grid = np.linspace(0.05, 3.0, 100)

    best_t, best_d_ece = 1.0, np.inf
    for t in t_grid:
        calibrated = apply_temperature(confidences, t)
        d_ece = compute_d_ece(calibrated, corrects, ious=ious, n_bins=n_bins)
        if d_ece < best_d_ece:
            best_d_ece, best_t = d_ece, t

    return best_t

def apply_temperature(confidences, T):
    eps = 1e-6
    conf_clipped = np.clip(confidences, eps, 1 - eps)
    logits = np.log(conf_clipped / (1 - conf_clipped))
    return 1 / (1 + np.exp(-logits / T))

def run_calibration_pipeline(val_dir, label_dir, val_sample_size=150, test_sample_size=293,
                              seed=42, conf_threshold=0.25, iou_threshold=0.5):
    print("Collecting validation-split (confidence, correct) pairs for fitting T...")
    val_conf, val_correct, val_ious = collect_confidence_correctness(
        val_dir, label_dir, sample_size=val_sample_size, seed=seed,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    print(f"Validation split: {len(val_conf)} detections collected.")

    T = fit_temperature(val_conf, val_correct)
    print(f"Fitted temperature T (NLL-targeted) = {T:.4f}")

    T_dece = fit_temperature_d_ece(val_conf, val_correct, val_ious)
    print(f"Fitted temperature T_dece (D-ECE-targeted) = {T_dece:.4f}")

    print("\nCollecting test-split (confidence, correct) pairs for before/after reporting...")
    test_conf, test_correct, test_ious = collect_confidence_correctness(
        val_dir, label_dir, sample_size=test_sample_size, seed=seed + 1,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    print(f"Test split: {len(test_conf)} detections collected.")

    ece_before = compute_ece(test_conf, test_correct)
    d_ece_before = compute_d_ece(test_conf, test_correct, ious=test_ious)

    test_conf_calibrated = apply_temperature(test_conf, T)
    ece_after = compute_ece(test_conf_calibrated, test_correct)
    d_ece_after = compute_d_ece(test_conf_calibrated, test_correct, ious=test_ious)

    test_conf_calibrated_dece = apply_temperature(test_conf, T_dece)
    ece_after_dece_target = compute_ece(test_conf_calibrated_dece, test_correct)
    d_ece_after_dece_target = compute_d_ece(test_conf_calibrated_dece, test_correct, ious=test_ious)

    print(f"\n=== CALIBRATION RESULTS ===")
    print(f"Temperature T (NLL-targeted, fit on validation split, frozen): {T:.4f}")
    print(f"ECE   before calibration: {ece_before:.4f}   after (T): {ece_after:.4f}")
    print(f"D-ECE before calibration: {d_ece_before:.4f}   after (T): {d_ece_after:.4f}")
    print(f"\nTemperature T_dece (D-ECE-targeted, fit on validation split, frozen): {T_dece:.4f}")
    print(f"ECE   after (T_dece): {ece_after_dece_target:.4f}")
    print(f"D-ECE after (T_dece): {d_ece_after_dece_target:.4f}")
    print(f"\nIf T and T_dece differ meaningfully and each metric improves most under its")
    print(f"OWN targeted temperature, this confirms binary-correctness calibration (ECE)")
    print(f"and IoU-weighted quality calibration (D-ECE) are genuinely different objectives")
    print(f"-- not a bug in either metric -- and a single T cannot simultaneously optimize both.")

    reliability_data = {
        "T": T, "T_dece": T_dece,
        "test_confidences_raw": test_conf,
        "test_confidences_calibrated": test_conf_calibrated,
        "test_confidences_calibrated_dece": test_conf_calibrated_dece,
        "test_corrects": test_correct,
        "test_ious": test_ious,
        "ece_before": ece_before, "ece_after": ece_after,
        "d_ece_before": d_ece_before, "d_ece_after": d_ece_after,
        "ece_after_dece_target": ece_after_dece_target,
        "d_ece_after_dece_target": d_ece_after_dece_target,
    }
    return reliability_data

def compute_mapr_calibrated(cls_name, raw_confidence, exposure, uniqueness, context, T,
                             weights=(0.3, 0.25, 0.2, 0.15, 0.1)):
    calibrated_confidence = apply_temperature(np.array([raw_confidence]), T)[0]
    sensitivity = SENSITIVITY_TABLE.get(cls_name, 0.3)
    w1, w2, w3, w4, w5 = weights
    return (w1 * sensitivity + w2 * calibrated_confidence + w3 * exposure
            + w4 * uniqueness + w5 * context)


