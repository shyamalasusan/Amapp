
import os
import random

import cv2
import numpy as np
import pandas as pd

from .baselines import baseline_fixed_mask_obb, baseline_global_blur, baseline_single_attribute_obb
from .config import DOTA_CLASSES, SENSITIVITY_TABLE  # noqa: F401  (re-exported for convenience)
from .metrics import compute_residual_detectability_obb, compute_utility_metrics
from .models import get_model_dota
from .pipeline_whole_object import run_amapp_linear_obb
from .scoring import obb_corners_to_aabb

# PART G — DOTA WHOLE-OBJECT EVALUATION (Table 2/3, linear)
# =============================================================================

def run_dota_whole_object_linear_eval(val_dir, sample_size=293, seed=42, conf_threshold=0.4):
    import pandas as pd
    random.seed(seed)
    all_val_images = os.listdir(val_dir)
    actual_sample_size = min(sample_size, len(all_val_images))
    sample_images = random.sample(all_val_images, actual_sample_size)
    print(f"DOTA val set has {len(all_val_images)} images; evaluating on {actual_sample_size}")

    results_log = {name: {"mae": [], "psnr": [], "ssim": [], "residual_detect": []}
                    for name in ["original", "global_blur", "fixed_mask", "single_attribute", "amapp_linear"]}
    skipped_no_detections = 0

    for idx, fname in enumerate(sample_images):
        image_path = os.path.join(val_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_shape = image.shape

        results = get_model_dota().predict(source=image_path, save=False, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            obb = r.obb
            if obb is None or len(obb) == 0:
                continue
            for i in range(len(obb)):
                cls_id = int(obb.cls[i])
                cls_name = DOTA_CLASSES[cls_id]
                confidence = float(obb.conf[i])
                corners = obb.xyxyxyxy[i].tolist()
                aabb = obb_corners_to_aabb(corners)
                detections.append((cls_name, confidence, aabb, corners))

        if len(detections) == 0:
            skipped_no_detections += 1
            continue

        original_count = len(detections)
        methods = {
            "original": image.copy(),
            "global_blur": baseline_global_blur(image.copy()),
            "fixed_mask": baseline_fixed_mask_obb(image, detections),
            "single_attribute": baseline_single_attribute_obb(image, detections),
        }
        methods["amapp_linear"], _ = run_amapp_linear_obb(image, detections, img_shape)

        for method_name, protected_img in methods.items():
            if method_name == "original":
                results_log[method_name]["mae"].append(0.0)
                results_log[method_name]["ssim"].append(1.0)
            else:
                mae, psnr_val, ssim_val = compute_utility_metrics(image, protected_img)
                results_log[method_name]["mae"].append(mae)
                results_log[method_name]["psnr"].append(psnr_val)
                results_log[method_name]["ssim"].append(ssim_val)
            residual = compute_residual_detectability_obb(protected_img, get_model_dota(), original_count)
            results_log[method_name]["residual_detect"].append(residual)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(sample_images)} images...")

    print(f"\nDone. Skipped {skipped_no_detections} images with zero detections.")

    table2_dota_rows, table3_dota_rows = [], []
    for method_name, metrics in results_log.items():
        n = len(metrics["mae"])
        if n == 0:
            continue
        table2_dota_rows.append({
            "Method": method_name,
            "Residual Detectability (mean)": np.mean(metrics["residual_detect"]),
            "Residual Detectability (std)": np.std(metrics["residual_detect"]),
            "Risk Reduction": 1 - np.mean(metrics["residual_detect"]),
            "N images": n,
        })
        psnr_mean = "N/A" if method_name == "original" else np.mean(metrics["psnr"])
        table3_dota_rows.append({
            "Method": method_name,
            "PSNR (mean)": psnr_mean,
            "SSIM (mean)": np.mean(metrics["ssim"]),
            "MAE (mean)": np.mean(metrics["mae"]),
        })

    table2_dota = pd.DataFrame(table2_dota_rows)
    table3_dota = pd.DataFrame(table3_dota_rows)

    print("\n=== DOTA TABLE 2: Privacy Effectiveness (oriented objects) ===")
    print(table2_dota.to_string(index=False))
    print("\n=== DOTA TABLE 3: Utility Preservation (oriented objects) ===")
    print(table3_dota.to_string(index=False))

    table2_dota.to_csv('/kaggle/working/table2_privacy_dota.csv', index=False)
    table3_dota.to_csv('/kaggle/working/table3_utility_dota.csv', index=False)
    print("\nSaved DOTA tables.")
    return table2_dota, table3_dota

