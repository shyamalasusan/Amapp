
import cv2

from .operators import apply_operator_obb

# PART D — BASELINE PROTECTION METHODS
# =============================================================================

def baseline_global_blur(image):
    return cv2.GaussianBlur(image, (51, 51), 0)

def baseline_fixed_mask_obb(image, detections):
    protected = image.copy()
    for cls_name, conf, aabb, corners in detections:
        protected = apply_operator_obb(protected, corners, "mask")
    return protected

def baseline_single_attribute_obb(image, detections):
    protected = image.copy()
    if not detections:
        return protected
    top = max(detections, key=lambda d: d[1])
    protected = apply_operator_obb(protected, top[3], "mask")
    return protected

