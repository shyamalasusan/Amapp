
import os
import random

import numpy as np
import pandas as pd

from .baselines import baseline_fixed_mask_obb, baseline_global_blur, baseline_single_attribute_obb
from .config import DIOR_CLASSES, DOTA_CLASSES
from .metrics import compute_residual_detectability, compute_residual_detectability_obb, compute_utility_metrics
from .models import get_model_dior, get_model_dota
from .pipeline_subregion import run_amapp_subregion_linear, run_amapp_subregion_linear_obb

# =============================================================================
# PART O — DIOR + DOTA SUB-REGION EVALUATION (Table 2/3)
# =============================================================================

def run_dior_subregion_eval(val_dir, sample_size=293, seed=42, conf_threshold=0.4):
    sam_model = get_sam_model()
    random.seed(seed)
    all_images = os.listdir(val_dir)
    sample_images = random.sample(all_images, min(sample_size, len(all_images)))
    print(f"DIOR val set has {len(all_images)} images; evaluating on {len(sample_images)}")

    results_log = {name: {"mae": [], "psnr": [], "ssim": [], "residual_detect": []}
                    for name in ["original", "global_blur", "fixed_mask",
                                 "single_attribute", "amapp_subregion_linear"]}
    subregion_path_count, fallback_count, total_eligible = 0, 0, 0
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
        methods = {
            "original": image.copy(),
            "global_blur": baseline_global_blur(image.copy()),
            "fixed_mask": baseline_fixed_mask(image, detections),
            "single_attribute": baseline_single_attribute(image, detections),
        }
        protected, per_detection_log = run_amapp_subregion_linear(image, detections, img_shape, sam_model)
        methods["amapp_subregion_linear"] = protected

        for entry in per_detection_log:
            if entry["class"] in SUBREGION_CLASSES:
                total_eligible += 1
                if entry["path"] == "subregion":
                    subregion_path_count += 1
                elif entry["path"] == "whole_object_fallback_no_masks":
                    fallback_count += 1

        for method_name, protected_img in methods.items():
            if method_name == "original":
                results_log[method_name]["mae"].append(0.0)
                results_log[method_name]["ssim"].append(1.0)
            else:
                mae, psnr_val, ssim_val = compute_utility_metrics(image, protected_img)
                results_log[method_name]["mae"].append(mae)
                results_log[method_name]["psnr"].append(psnr_val)
                results_log[method_name]["ssim"].append(ssim_val)
            residual = compute_residual_detectability(protected_img, get_model_dior(), DIOR_CLASSES, original_count)
            results_log[method_name]["residual_detect"].append(residual)

        if (idx + 1) % 25 == 0:
            print(f"[DIOR] Processed {idx + 1}/{len(sample_images)} images...")
        elif (idx + 1) % 5 == 0:
            print(f"  [DIOR] ...{idx + 1}/{len(sample_images)} "
                  f"(sub-region path used on {subregion_path_count}/{total_eligible} eligible detections)")

    print(f"\n[DIOR] Done. Skipped {skipped_no_detections} images with zero detections.")
    print(f"[DIOR] Sub-region-eligible detections: {total_eligible} total, "
          f"{subregion_path_count} via sub-region path, {fallback_count} fell back to whole-object.")
    return results_log

def run_dota_subregion_eval(val_dir, sample_size=293, seed=42, conf_threshold=0.4):
    sam_model = get_sam_model()
    random.seed(seed)
    all_images = os.listdir(val_dir)
    sample_images = random.sample(all_images, min(sample_size, len(all_images)))
    print(f"DOTA val set has {len(all_images)} images; evaluating on {len(sample_images)}")

    results_log = {name: {"mae": [], "psnr": [], "ssim": [], "residual_detect": []}
                    for name in ["original", "global_blur", "fixed_mask",
                                 "single_attribute", "amapp_subregion_linear"]}
    subregion_path_count, fallback_count, total_eligible = 0, 0, 0
    skipped_no_detections = 0

    for idx, fname in enumerate(sample_images):
        image_path = os.path.join(val_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_shape = image.shape

        results = get_model_dota().predict(source=image_path, save=False,
                                      conf=conf_threshold, verbose=False)
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

        if not detections:
            skipped_no_detections += 1
            continue

        original_count = len(detections)
        methods = {
            "original": image.copy(),
            "global_blur": baseline_global_blur(image.copy()),
            "fixed_mask": baseline_fixed_mask_obb(image, detections),
            "single_attribute": baseline_single_attribute_obb(image, detections),
        }
        protected, per_detection_log = run_amapp_subregion_linear_obb(image, detections, img_shape, sam_model)
        methods["amapp_subregion_linear"] = protected

        for entry in per_detection_log:
            if entry["class"] in SUBREGION_CLASSES:
                total_eligible += 1
                if entry["path"] == "subregion":
                    subregion_path_count += 1
                elif entry["path"] == "whole_object_fallback_no_masks":
                    fallback_count += 1

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
            print(f"[DOTA] Processed {idx + 1}/{len(sample_images)} images...")
        elif (idx + 1) % 5 == 0:
            print(f"  [DOTA] ...{idx + 1}/{len(sample_images)} "
                  f"(sub-region path used on {subregion_path_count}/{total_eligible} eligible detections)")

    print(f"\n[DOTA] Done. Skipped {skipped_no_detections} images with zero detections.")
    print(f"[DOTA] Sub-region-eligible detections: {total_eligible} total, "
          f"{subregion_path_count} via sub-region path, {fallback_count} fell back to whole-object.")
    return results_log

def summarize_and_save(results_log, dataset_label, out_prefix):

    table2_rows, table3_rows = [], []
    for method_name, metrics in results_log.items():
        n = len(metrics["mae"])
        if n == 0:
            continue
        table2_rows.append({
            "Method": method_name,
            "Residual Detectability (mean)": np.mean(metrics["residual_detect"]),
            "Residual Detectability (std)": np.std(metrics["residual_detect"]),
            "Risk Reduction": 1 - np.mean(metrics["residual_detect"]),
            "N images": n,
        })
        psnr_mean = "N/A" if method_name == "original" else np.mean(metrics["psnr"])
        table3_rows.append({
            "Method": method_name,
            "PSNR (mean)": psnr_mean,
            "SSIM (mean)": np.mean(metrics["ssim"]),
            "MAE (mean)": np.mean(metrics["mae"]),
        })

    table2 = pd.DataFrame(table2_rows)
    table3 = pd.DataFrame(table3_rows)

    print(f"\n=== {dataset_label} SUB-REGION (LINEAR) TABLE 2: Privacy Effectiveness ===")
    print(table2.to_string(index=False))
    print(f"\n=== {dataset_label} SUB-REGION (LINEAR) TABLE 3: Utility Preservation ===")
    print(table3.to_string(index=False))

    table2.to_csv(f'/kaggle/working/{out_prefix}_table2_privacy.csv', index=False)
    table3.to_csv(f'/kaggle/working/{out_prefix}_table3_utility.csv', index=False)
    print(f"\nSaved {dataset_label} sub-region (linear) tables.")
    return table2, table3


