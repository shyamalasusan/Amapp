
from .config import SENSITIVITY_TABLE

# PART C — MAPR RISK SCORING (linear formula)
# =============================================================================

def obb_corners_to_aabb(corners):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return [min(xs), min(ys), max(xs), max(ys)]

def compute_exposure(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    img_h, img_w = image_shape[:2]
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    clipped_x1, clipped_y1 = max(0, x1), max(0, y1)
    clipped_x2, clipped_y2 = min(img_w, x2), min(img_h, y2)
    clipped_area = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)
    if box_area == 0:
        return 0.0
    return clipped_area / box_area

def compute_uniqueness(cls_name, all_detections_in_image):
    class_counts = {}
    for d in all_detections_in_image:
        class_counts[d[0]] = class_counts.get(d[0], 0) + 1
    if not class_counts:
        return 0.5
    max_count = max(class_counts.values())
    this_count = class_counts.get(cls_name, 1)
    return 1 - (this_count / max_count) if max_count > 0 else 0.5

def compute_context(bbox, all_detections_in_image, radius=200):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nearby_risk = 0.0
    for cls_name, conf, other_bbox in all_detections_in_image:
        ox1, oy1, ox2, oy2 = other_bbox
        ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
        if 0 < dist <= radius:
            nearby_risk += SENSITIVITY_TABLE.get(cls_name, 0.3)
    return min(1.0, nearby_risk / 3.0)

def compute_mapr(cls_name, confidence, exposure, uniqueness, context,
                  weights=(0.3, 0.25, 0.2, 0.15, 0.1)):
    w1, w2, w3, w4, w5 = weights
    sensitivity = SENSITIVITY_TABLE.get(cls_name, 0.3)
    return (w1 * sensitivity + w2 * confidence + w3 * exposure
            + w4 * uniqueness + w5 * context)


def select_operator(mapr_score):
    if mapr_score < 0.25: return "none"
    elif mapr_score < 0.5: return "blur"
    elif mapr_score < 0.75: return "pixelate"
    else: return "mask"

