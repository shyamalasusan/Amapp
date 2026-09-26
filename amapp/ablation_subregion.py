

import os
import random

import numpy as np
import pandas as pd

from .ablation_config import ABLATION_CONFIGS, CONFIG_DISPLAY_NAMES
from .config import DIOR_CLASSES, SENSITIVITY_TABLE, SENSITIVITY_TABLE_PARTS, SUBREGION_CLASSES
from .metrics import compute_residual_detectability
from .models import get_model_dior
from .part_labeling import label_parts
from .segmentation import get_sam_model, segment_detections_with_sam_batched
from .subregion_scoring import apply_operator_multiregion

# PART T — TABLE 4: SUB-REGION ABLATION
# =============================================================================

def get_part_sensitivity_ablation(class_name, part_name):
    return get_part_sensitivity(class_name, part_name)

def compute_mapr_part_ablation(part_sensitivity, confidence, exposure, uniqueness, context, config):
    if config == 'sensitivity_only':
        return part_sensitivity
    elif config == 'plus_confidence':
        return 0.5 * part_sensitivity + 0.5 * confidence
    elif config == 'plus_exposure':
        return 0.4 * part_sensitivity + 0.3 * confidence + 0.3 * exposure
    elif config == 'full':
        return compute_mapr_part_linear(part_sensitivity, confidence, exposure, uniqueness, context)
    else:
        raise ValueError(f"Unknown config: {config}")

def score_all_parts_ablation(class_name, labeled_parts, confidence, exposure,
                              uniqueness, context, config):
    scored = []
    for part_name, mask in labeled_parts:
        part_sensitivity = get_part_sensitivity_ablation(class_name, part_name)
        mapr = compute_mapr_part_ablation(part_sensitivity, confidence, exposure,
                                           uniqueness, context, config)
        operator = select_operator(mapr)
        scored.append((part_name, mask, mapr, operator))
    return scored

def run_amapp_subregion_config(image, detections, img_shape, config, sam_model=None):
    if sam_model is None:
        sam_model = get_sam_model()

    protected = image.copy()
    per_detection_log = []

    subregion_indices = [
        i for i, (cls_name, _, _) in enumerate(detections)
        if cls_name in SUBREGION_CLASSES
    ]
    subregion_bboxes = [detections[i][2] for i in subregion_indices]
    batched_masks = segment_detections_with_sam_batched(protected, subregion_bboxes, sam_model)
    masks_by_index = dict(zip(subregion_indices, batched_masks))

    for det_idx, (cls_name, confidence, bbox) in enumerate(detections):
        exposure = compute_exposure(bbox, img_shape)
        uniqueness = compute_uniqueness(cls_name, detections)
        context = compute_context(bbox, detections)

        if cls_name in SUBREGION_CLASSES:
            raw_masks = masks_by_index.get(det_idx, [])
            labeled_parts = label_parts(raw_masks, bbox, cls_name)

            if not labeled_parts:
                mapr = compute_mapr_ablation(cls_name, confidence, exposure, uniqueness, context, config)
                operator = select_operator(mapr)
                protected = apply_operator(protected, bbox, operator)
                per_detection_log.append({"class": cls_name, "path": "whole_object_fallback_no_masks"})
                continue

            scored_parts = score_all_parts_ablation(cls_name, labeled_parts, confidence,
                                                     exposure, uniqueness, context, config)
            protected = apply_operator_multiregion(protected, scored_parts)
            per_detection_log.append({"class": cls_name, "path": "subregion"})
        else:
            mapr = compute_mapr_ablation(cls_name, confidence, exposure, uniqueness, context, config)
            operator = select_operator(mapr)
            protected = apply_operator(protected, bbox, operator)
            per_detection_log.append({"class": cls_name, "path": "whole_object"})

    return protected, per_detection_log

def run_table4_subregion_ablation(val_dir, sample_size=293, seed=42, conf_threshold=0.4):
    sam_model = get_sam_model()
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
            protected, _ = run_amapp_subregion_config(image, detections, img_shape, config, sam_model)
            _, psnr_val, _ = compute_utility_metrics(image, protected)
            residual = compute_residual_detectability(protected, get_model_dior(), DIOR_CLASSES, original_count)
            results_log[config]["psnr"].append(psnr_val)
            results_log[config]["residual_detect"].append(residual)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(sample_images)} images...")
        elif (idx + 1) % 5 == 0:
            print(f"  ...{idx + 1}/{len(sample_images)}")

    print(f"\nDone. Skipped {skipped_no_detections} images with zero detections.")
    return results_log

def summarize_table4_subregion(results_log):

    rows = []
    for config in ABLATION_CONFIGS:
        metrics = results_log[config]
        n = len(metrics["psnr"])
        if n == 0:
            continue
        mean_residual = np.mean(metrics["residual_detect"])
        risk_reduction = 1 - mean_residual
        mean_psnr = np.mean(metrics["psnr"])
        normalized_psnr = min(mean_psnr / PSNR_NORMALIZATION_CAP, 1.0) if np.isfinite(mean_psnr) else 1.0
        privacy_utility_score = 0.5 * risk_reduction + 0.5 * normalized_psnr

        rows.append({
            "Configuration": CONFIG_DISPLAY_NAMES[config],
            "Risk Reduction": risk_reduction,
            "PSNR (mean)": mean_psnr,
            "Privacy-utility score": privacy_utility_score,
            "N images": n,
        })

    table4_sr = pd.DataFrame(rows)
    print("\n=== TABLE 4 (MANUSCRIPT, SUB-REGION): Component Ablation ===")
    print(table4_sr.to_string(index=False))
    table4_sr.to_csv('/kaggle/working/table4_subregion_component_ablation.csv', index=False)
    print("\nSaved table4_subregion_component_ablation.csv")
    print("\nSANITY CHECK: the 'Full MAPR (linear)' row here should be close to")
    print("this project's existing sub-region Table 2/3 numbers (68.4% risk")
    print("reduction, PSNR 30.21) -- both use compute_mapr_part_linear() and the")
    print("same seed=42 sample.")
    return table4_sr


