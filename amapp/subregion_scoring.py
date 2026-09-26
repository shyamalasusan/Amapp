

import cv2
import numpy as np

from .config import SENSITIVITY_TABLE, SENSITIVITY_TABLE_PARTS

# PART K — SUB-REGION MAPR SCORING + MULTI-REGION PROTECTION
# =============================================================================

def get_part_sensitivity(class_name, part_name):
    class_parts = SENSITIVITY_TABLE_PARTS.get(class_name, {})
    if part_name in class_parts:
        return class_parts[part_name]
    return SENSITIVITY_TABLE.get(class_name, 0.3)

def compute_mapr_part_linear(part_sensitivity, confidence, exposure, uniqueness,
                              context, weights=(0.3, 0.25, 0.2, 0.15, 0.1)):
    w1, w2, w3, w4, w5 = weights
    return (w1 * part_sensitivity + w2 * confidence + w3 * exposure
            + w4 * uniqueness + w5 * context)

def score_all_parts(class_name, labeled_parts, confidence, exposure,
                     uniqueness, context):
    scored = []
    for part_name, mask in labeled_parts:
        part_sensitivity = get_part_sensitivity(class_name, part_name)
        mapr = compute_mapr_part_linear(part_sensitivity, confidence, exposure,
                                         uniqueness, context)
        operator = select_operator(mapr)
        scored.append((part_name, mask, mapr, operator))
    return scored

def apply_operator_to_mask(image, mask, operator):
    if operator in ("preserve", "none") or not mask.any():
        return image

    protected = image.copy()
    ys, xs = np.where(mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    if operator == "blur":
        transformed_roi = cv2.GaussianBlur(roi, (31, 31), 0)
    elif operator == "pixelate":
        small = cv2.resize(roi, (max(roi.shape[1] // 15, 1), max(roi.shape[0] // 15, 1)),
                            interpolation=cv2.INTER_LINEAR)
        transformed_roi = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
    elif operator == "mask":
        transformed_roi = np.zeros_like(roi)
    else:
        raise ValueError(f"Unknown operator: {operator}")

    roi_out = roi.copy()
    roi_out[roi_mask] = transformed_roi[roi_mask]
    protected[y1:y2, x1:x2] = roi_out
    return protected

def apply_operator_multiregion(image, scored_parts):
    protected = image.copy()
    for part_name, mask, mapr, operator in sorted(scored_parts, key=lambda p: p[2], reverse=True):
        protected = apply_operator_to_mask(protected, mask, operator)
    return protected


