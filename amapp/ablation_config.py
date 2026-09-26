

import os
import random

import numpy as np
import pandas as pd

from .config import DIOR_CLASSES, SENSITIVITY_TABLE
from .metrics import compute_residual_detectability
from .models import get_model_dior
from .operators import apply_operator
from .scoring import compute_context, compute_exposure, compute_uniqueness, select_operator

# PART R — TABLE 4 HELPERS (component configs, privacy-utility score)
# =============================================================================

PSNR_NORMALIZATION_CAP = 35.0  # see docstring above for justification

def compute_mapr_ablation(cls_name, confidence, exposure, uniqueness, context, config):
    sensitivity = SENSITIVITY_TABLE.get(cls_name, 0.3)
    if config == 'sensitivity_only':
        return sensitivity
    elif config == 'plus_confidence':
        return 0.5 * sensitivity + 0.5 * confidence
    elif config == 'plus_exposure':
        return 0.4 * sensitivity + 0.3 * confidence + 0.3 * exposure
    elif config == 'full':
        return compute_mapr(cls_name, confidence, exposure, uniqueness, context)
    else:
        raise ValueError(f"Unknown config: {config}")

ABLATION_CONFIGS = ['sensitivity_only', 'plus_confidence', 'plus_exposure', 'full']

CONFIG_DISPLAY_NAMES = {
    'sensitivity_only': 'Sensitivity only',
    'plus_confidence': '+ Confidence',
    'plus_exposure': '+ Exposure',
    'full': 'Full MAPR (linear)',
}

def run_amapp_config(image, detections, img_shape, config):
    protected = image.copy()
    scored = []
    for cls_name, confidence, bbox in detections:
        exposure = compute_exposure(bbox, img_shape)
        uniqueness = compute_uniqueness(cls_name, detections)
        context = compute_context(bbox, detections)
        mapr = compute_mapr_ablation(cls_name, confidence, exposure, uniqueness, context, config)
        scored.append((cls_name, confidence, bbox, mapr))
    scored.sort(key=lambda d: d[3], reverse=True)
    for cls_name, confidence, bbox, mapr in scored:
        operator = select_operator(mapr)
        protected = apply_operator(protected, bbox, operator)
    return protected, scored

def run_table4_component_ablation(val_dir, sample_size=293, seed=42, conf_threshold=0.4):
    random.seed(seed)
    all_images = os.listdir(val_dir)
    sample_images = random.sample(all_images, min(sample_size, len(all_images)))
    print(f"DIOR val set has {len(all_images)} images; evaluating on {len(sample_images)}")

    results_log = {config: {"psnr": [], "residual_detect": []} for config in ABLATION_CONFIGS}
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
        for config in ABLATION_CONFIGS:
            protected, _ = run_amapp_config(image, detections, img_shape, config)
            _, psnr_val, _ = compute_utility_metrics(image, protected)
            residual = compute_residual_detectability(protected, get_model_dior(), DIOR_CLASSES, original_count)
            results_log[config]["psnr"].append(psnr_val)
            results_log[config]["residual_detect"].append(residual)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(sample_images)} images...")

    print(f"\nDone. Skipped {skipped_no_detections} images with zero detections.")
    return results_log

def summarize_table4(results_log):

    rows = []
    for config in ABLATION_CONFIGS:
        metrics = results_log[config]
        n = len(metrics["psnr"])
        if n == 0:
            continue
        mean_residual = np.mean(metrics["residual_detect"])
        risk_reduction = 1 - mean_residual
        mean_psnr = np.mean(metrics["psnr"])
        normalized_psnr = min(mean_psnr / PSNR_NORMALIZATION_CAP, 1.0)
        privacy_utility_score = 0.5 * risk_reduction + 0.5 * normalized_psnr

        rows.append({
            "Configuration": CONFIG_DISPLAY_NAMES[config],
            "Risk Reduction": risk_reduction,
            "PSNR (mean)": mean_psnr,
            "Privacy-utility score": privacy_utility_score,
            "N images": n,
        })

    table4 = pd.DataFrame(rows)
    print("\n=== TABLE 4 (MANUSCRIPT): Component Ablation ===")
    print(table4.to_string(index=False))
    table4.to_csv('/kaggle/working/table4_component_ablation.csv', index=False)
    print("\nSaved table4_component_ablation.csv")
    print("\nSANITY CHECK: the 'Full MAPR (linear)' row's Risk Reduction and PSNR")
    print("should closely match this project's existing DIOR Table 2/3 whole-object")
    print("linear results (74.8% risk reduction, PSNR 29.82) -- both use the same")
    print("compute_mapr() formula and the same seed=42 sample. Small differences")
    print("are expected only if the detector or conf_threshold changed since then.")
    return table4


