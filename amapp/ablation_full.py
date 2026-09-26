


import os
import random
import time

import numpy as np
import pandas as pd

from .ablation_config import ABLATION_CONFIGS, CONFIG_DISPLAY_NAMES, PSNR_NORMALIZATION_CAP
from .config import DIOR_CLASSES, SENSITIVITY_TABLE
from .metrics import compute_residual_detectability
from .models import get_model_dior
from .operators import apply_operator
from .scoring import compute_context, compute_exposure, compute_uniqueness, select_operator

# PART S — TABLE 4: A0-A5 (whole-object ablation + weight sensitivity + runtime)
# =============================================================================

def compute_mapr_a0_a4(cls_name, confidence, exposure, uniqueness, context, config,
                        weights=(0.3, 0.25, 0.2, 0.15, 0.1)):
    w_sens, w_conf, w_exp, w_uniq, w_ctx = weights
    sensitivity = SENSITIVITY_TABLE.get(cls_name, 0.3)

    if config == 'A0':
        return w_sens * sensitivity
    elif config == 'A1':
        return w_sens * sensitivity + w_conf * confidence
    elif config == 'A2':
        return w_sens * sensitivity + w_conf * confidence + w_exp * exposure
    elif config == 'A3':
        return (w_sens * sensitivity + w_conf * confidence + w_exp * exposure
                + w_uniq * uniqueness)
    elif config == 'A4':
        return compute_mapr(cls_name, confidence, exposure, uniqueness, context, weights)
    else:
        raise ValueError(f"Unknown A0-A4 config: {config}")

A5_WEIGHT_PERTURBATIONS = {
    'A5_baseline':        (0.30, 0.25, 0.20, 0.15, 0.10),  # = A4, included for direct comparison
    'A5_sens_up':         (0.45, 0.20, 0.15, 0.12, 0.08),  # sensitivity weighted more heavily
    'A5_sens_down':       (0.15, 0.30, 0.25, 0.18, 0.12),  # sensitivity weighted less heavily
    'A5_conf_up':         (0.25, 0.40, 0.15, 0.12, 0.08),  # confidence weighted more heavily
    'A5_equal':           (0.20, 0.20, 0.20, 0.20, 0.20),  # all five components equal
}

A0_A4_CONFIGS = ['A0', 'A1', 'A2', 'A3', 'A4']

A0_A4_DISPLAY_NAMES = {
    'A0': 'A0: Sensitivity only',
    'A1': 'A1: + Confidence',
    'A2': 'A2: + Exposure',
    'A3': 'A3: + Uniqueness',
    'A4': 'A4: Full MAPR (+ Context)',
}

def run_amapp_a0_a4(image, detections, img_shape, config):
    protected = image.copy()
    scored = []
    for cls_name, confidence, bbox in detections:
        exposure = compute_exposure(bbox, img_shape)
        uniqueness = compute_uniqueness(cls_name, detections)
        context = compute_context(bbox, detections)
        mapr = compute_mapr_a0_a4(cls_name, confidence, exposure, uniqueness, context, config)
        scored.append((cls_name, confidence, bbox, mapr))
    scored.sort(key=lambda d: d[3], reverse=True)
    for cls_name, confidence, bbox, mapr in scored:
        operator = select_operator(mapr)
        protected = apply_operator(protected, bbox, operator)
    return protected, scored

def run_amapp_a5(image, detections, img_shape, weight_vector):
    protected = image.copy()
    scored = []
    for cls_name, confidence, bbox in detections:
        exposure = compute_exposure(bbox, img_shape)
        uniqueness = compute_uniqueness(cls_name, detections)
        context = compute_context(bbox, detections)
        mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context, weight_vector)
        scored.append((cls_name, confidence, bbox, mapr))
    scored.sort(key=lambda d: d[3], reverse=True)
    for cls_name, confidence, bbox, mapr in scored:
        operator = select_operator(mapr)
        protected = apply_operator(protected, bbox, operator)
    return protected, scored

def run_table4_full_ablation(val_dir, sample_size=293, seed=42, conf_threshold=0.4):
    random.seed(seed)
    all_images = os.listdir(val_dir)
    sample_images = random.sample(all_images, min(sample_size, len(all_images)))
    print(f"DIOR val set has {len(all_images)} images; evaluating on {len(sample_images)}")

    all_configs = A0_A4_CONFIGS + list(A5_WEIGHT_PERTURBATIONS.keys())
    results_log = {config: {"psnr": [], "residual_detect": [], "runtime_sec": []}
                    for config in all_configs}
    skipped_no_detections = 0

    for idx, fname in enumerate(sample_images):
        image_path = os.path.join(val_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_shape = image.shape

        results = get_model_dior().predict(source=image_path, save=False,
                                      conf=conf_threshold, verbose=False)
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

        if not detections:
            skipped_no_detections += 1
            continue

        original_count = len(detections)

        for config in A0_A4_CONFIGS:
            t0 = time.perf_counter()
            protected, _ = run_amapp_a0_a4(image, detections, img_shape, config)
            runtime = time.perf_counter() - t0
            _, psnr_val, _ = compute_utility_metrics(image, protected)
            residual = compute_residual_detectability(protected, get_model_dior(), DIOR_CLASSES, original_count)
            results_log[config]["psnr"].append(psnr_val)
            results_log[config]["residual_detect"].append(residual)
            results_log[config]["runtime_sec"].append(runtime)

        for name, weight_vector in A5_WEIGHT_PERTURBATIONS.items():
            t0 = time.perf_counter()
            protected, _ = run_amapp_a5(image, detections, img_shape, weight_vector)
            runtime = time.perf_counter() - t0
            _, psnr_val, _ = compute_utility_metrics(image, protected)
            residual = compute_residual_detectability(protected, get_model_dior(), DIOR_CLASSES, original_count)
            results_log[name]["psnr"].append(psnr_val)
            results_log[name]["residual_detect"].append(residual)
            results_log[name]["runtime_sec"].append(runtime)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(sample_images)} images...")

    print(f"\nDone. Skipped {skipped_no_detections} images with zero detections.")
    return results_log

def summarize_table4_full(results_log):

    def _row(display_name, metrics):
        n = len(metrics["psnr"])
        if n == 0:
            return None
        mean_residual = np.mean(metrics["residual_detect"])
        risk_reduction = 1 - mean_residual
        mean_psnr = np.mean(metrics["psnr"])
        normalized_psnr = min(mean_psnr / PSNR_NORMALIZATION_CAP, 1.0) if np.isfinite(mean_psnr) else 1.0
        privacy_utility_score = 0.5 * risk_reduction + 0.5 * normalized_psnr
        mean_runtime_ms = np.mean(metrics["runtime_sec"]) * 1000
        return {
            "Configuration": display_name,
            "Risk Reduction": risk_reduction,
            "PSNR (mean)": mean_psnr,
            "Privacy-utility score": privacy_utility_score,
            "Runtime (ms/image, mean)": mean_runtime_ms,
            "N images": n,
        }

    a0_a4_rows = [_row(A0_A4_DISPLAY_NAMES[c], results_log[c]) for c in A0_A4_CONFIGS]
    a0_a4_rows = [r for r in a0_a4_rows if r is not None]
    table4_a0a4 = pd.DataFrame(a0_a4_rows)

    a5_rows = [_row(name, results_log[name]) for name in A5_WEIGHT_PERTURBATIONS]
    a5_rows = [r for r in a5_rows if r is not None]
    table4_a5 = pd.DataFrame(a5_rows)

    print("\n=== TABLE 4a (A0-A4): Component Ablation ===")
    print(table4_a0a4.to_string(index=False))
    print("\n=== TABLE 4b (A5): Weight/Threshold Sensitivity, Full MAPR ===")
    print(table4_a5.to_string(index=False))

    table4_a0a4.to_csv('/kaggle/working/table4a_component_ablation_full.csv', index=False)
    table4_a5.to_csv('/kaggle/working/table4b_weight_sensitivity.csv', index=False)
    print("\nSaved table4a_component_ablation_full.csv and table4b_weight_sensitivity.csv")

    print("\nSANITY CHECKS:")
    print("- A4's Risk Reduction/PSNR should be close to this project's existing")
    print("  DIOR Table 2/3 whole-object linear results (74.8%, PSNR 29.82).")
    print("- A5_baseline uses the SAME weights as A4 -- its numbers should be")
    print("  IDENTICAL to A4's (this is a built-in consistency check, not an")
    print("  independent finding -- if they differ noticeably, something is wrong).")
    return table4_a0a4, table4_a5


