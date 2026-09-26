
import numpy as np
import cv2

from .config import SUBREGION_CLASSES
from .operators import apply_operator, apply_operator_obb
from .part_labeling import label_parts
from .scoring import compute_context, compute_exposure, compute_uniqueness, select_operator
from .segmentation import get_sam_model, segment_detection_with_sam, segment_detections_with_sam_batched
from .subregion_scoring import apply_operator_multiregion, score_all_parts

# PART L — SUB-REGION PIPELINE (DIOR)
# =============================================================================

def run_amapp_subregion_linear(image, detections, img_shape, sam_model=None):
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
                mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context)
                operator = select_operator(mapr)
                protected = apply_operator(protected, bbox, operator)
                per_detection_log.append({
                    "class": cls_name, "path": "whole_object_fallback_no_masks",
                    "mapr": mapr, "operator": operator,
                })
                continue

            scored_parts = score_all_parts(cls_name, labeled_parts, confidence,
                                            exposure, uniqueness, context)
            protected = apply_operator_multiregion(protected, scored_parts)
            per_detection_log.append({
                "class": cls_name, "path": "subregion",
                "parts": [(p[0], p[2], p[3]) for p in scored_parts],
            })
        else:
            mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context)
            operator = select_operator(mapr)
            protected = apply_operator(protected, bbox, operator)
            per_detection_log.append({
                "class": cls_name, "path": "whole_object",
                "mapr": mapr, "operator": operator,
            })

    return protected, per_detection_log



# PART N — SUB-REGION OBB SUPPORT (DOTA)
# =============================================================================

def polygon_to_mask(corners, img_shape):
    h, w = img_shape[:2]
    pts = np.array(corners, dtype=np.int32).reshape(-1, 2)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts], 1)
    return poly_mask.astype(bool)

def run_amapp_subregion_linear_obb(image, detections, img_shape, sam_model=None):
    if sam_model is None:
        sam_model = get_sam_model()

    protected = image.copy()
    per_detection_log = []

    subregion_indices = [
        i for i, (cls_name, _, _, _) in enumerate(detections)
        if cls_name in SUBREGION_CLASSES
    ]
    subregion_aabbs = [detections[i][2] for i in subregion_indices]
    batched_masks = segment_detections_with_sam_batched(protected, subregion_aabbs, sam_model)
    masks_by_index = dict(zip(subregion_indices, batched_masks))

    for det_idx, (cls_name, confidence, aabb, corners) in enumerate(detections):
        exposure = compute_exposure(aabb, img_shape)
        detections_view = [(d[0], d[1], d[2]) for d in detections]
        uniqueness = compute_uniqueness(cls_name, detections_view)
        context = compute_context(aabb, detections_view)

        if cls_name in SUBREGION_CLASSES:
            raw_masks = masks_by_index.get(det_idx, [])

            polygon_mask = polygon_to_mask(corners, img_shape)
            clipped_masks = []
            for mask, _ in raw_masks:
                clipped = mask & polygon_mask
                area = int(clipped.sum())
                if area > 0:
                    clipped_masks.append((clipped, area))
            clipped_masks.sort(key=lambda t: t[1], reverse=True)

            labeled_parts = label_parts(clipped_masks, aabb, cls_name)

            if not labeled_parts:
                mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context)
                operator = select_operator(mapr)
                protected = apply_operator_obb(protected, corners, operator)
                per_detection_log.append({
                    "class": cls_name, "path": "whole_object_fallback_no_masks",
                    "mapr": mapr, "operator": operator,
                })
                continue

            scored_parts = score_all_parts(cls_name, labeled_parts, confidence,
                                            exposure, uniqueness, context)
            protected = apply_operator_multiregion(protected, scored_parts)
            per_detection_log.append({
                "class": cls_name, "path": "subregion",
                "parts": [(p[0], p[2], p[3]) for p in scored_parts],
            })
        else:
            mapr = compute_mapr(cls_name, confidence, exposure, uniqueness, context)
            operator = select_operator(mapr)
            protected = apply_operator_obb(protected, corners, operator)
            per_detection_log.append({
                "class": cls_name, "path": "whole_object",
                "mapr": mapr, "operator": operator,
            })

    return protected, per_detection_log


