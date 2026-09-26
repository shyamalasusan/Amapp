
from .operators import apply_operator, apply_operator_obb
from .scoring import compute_context, compute_exposure, compute_mapr, compute_uniqueness, select_operator

# PART E — WHOLE-OBJECT AMAPP PIPELINE (DIOR + DOTA/OBB)
# =============================================================================

def run_amapp_linear_obb(image, detections, img_shape):
    protected = image.copy()
    scored = []
    for cls_name, confidence, aabb, corners in detections:
        exposure = compute_exposure(aabb, img_shape)
        detections_view = [(d[0], d[1], d[2]) for d in detections]
        uniqueness = compute_uniqueness(cls_name, detections_view)
        context = compute_context(aabb, detections_view)
        mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context)
        scored.append((cls_name, confidence, aabb, corners, mapr))
    scored.sort(key=lambda d: d[4], reverse=True)
    for cls_name, confidence, aabb, corners, mapr in scored:
        operator = select_operator(mapr)
        protected = apply_operator_obb(protected, corners, operator)
    return protected, scored

def run_amapp_linear(image, detections, img_shape):
    protected = image.copy()
    scored = []
    for cls_name, confidence, bbox in detections:
        exposure = compute_exposure(bbox, img_shape)
        uniqueness = compute_uniqueness(cls_name, detections)
        context = compute_context(bbox, detections)
        mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context)
        scored.append((cls_name, confidence, bbox, mapr))
    scored.sort(key=lambda d: d[3], reverse=True)
    for cls_name, confidence, bbox, mapr in scored:
        operator = select_operator(mapr)
        protected = apply_operator(protected, bbox, operator)
    return protected, scored

