
import cv2
import numpy as np
from ultralytics import SAM

# PART I — SAM SEGMENTATION (box-prompted, multimask)
# =============================================================================

_sam_wrapper = None

_sam_predictor = None

def get_sam_model(checkpoint="sam_b.pt"):
    # bboxes=[...], multimask_output=True)`, matching how
    global _sam_wrapper, _sam_predictor
    if _sam_predictor is None:
        _sam_wrapper = SAM(checkpoint)
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        _sam_wrapper.predict(dummy_image, bboxes=[[0, 0, 10, 10]], verbose=False)
        _sam_predictor = _sam_wrapper.predictor
    return _sam_predictor

def segment_detection_with_sam(image, bbox, sam_model=None):
    # internal parts. This function requests multimask_output=True via the
    # SyntaxError: 'multimask_output' is not a valid YOLO argument" -- the
    if sam_model is None:
        sam_model = get_sam_model()

    x1, y1, x2, y2 = [int(v) for v in bbox]
    results = sam_model(source=image, bboxes=[[x1, y1, x2, y2]], multimask_output=True)

    masks = []
    for r in results:
        if r.masks is None:
            continue
        for m in r.masks.data:
            mask_np = m.cpu().numpy().astype(bool)
            if mask_np.shape != image.shape[:2]:
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            area = int(mask_np.sum())
            if area > 0:
                masks.append((mask_np, area))

    masks = _dedupe_masks(masks)
    masks.sort(key=lambda t: t[1], reverse=True)
    return masks

def _dedupe_masks(masks, iou_threshold=0.9):
    # multimask_output=True can return near-duplicate masks (e.g. two of the
    kept = []
    for mask, area in sorted(masks, key=lambda t: t[1], reverse=True):
        is_duplicate = False
        for kept_mask, _ in kept:
            intersection = np.logical_and(mask, kept_mask).sum()
            union = np.logical_or(mask, kept_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((mask, area))
    return kept

def segment_detections_with_sam_batched(image, bboxes, sam_model=None):
    # Also applies Finding 3's multimask_output=True fix (see
    # concatenated in prompt order -- with multimask_output=True, that's
    if sam_model is None:
        sam_model = get_sam_model()
    if not bboxes:
        return []

    MASKS_PER_BOX = 3  # SAM's multimask_output=True candidate count

    int_boxes = [[int(v) for v in b] for b in bboxes]
    results = sam_model(source=image, bboxes=int_boxes, multimask_output=True)

    flat_masks = []
    for r in results:
        if r.masks is None:
            continue
        for m in r.masks.data:
            mask_np = m.cpu().numpy().astype(bool)
            if mask_np.shape != image.shape[:2]:
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            flat_masks.append(mask_np)

    expected_total = len(int_boxes) * MASKS_PER_BOX
    if len(flat_masks) != expected_total:
        per_box_masks = []
        for bbox in bboxes:
            per_box_masks.append(segment_detection_with_sam(image, bbox, sam_model))
        return per_box_masks

    per_box_masks = []
    for i in range(len(int_boxes)):
        box_masks_raw = flat_masks[i * MASKS_PER_BOX:(i + 1) * MASKS_PER_BOX]
        box_masks = [(m, int(m.sum())) for m in box_masks_raw if m.sum() > 0]
        box_masks = _dedupe_masks(box_masks)
        box_masks.sort(key=lambda t: t[1], reverse=True)
        per_box_masks.append(box_masks)
    return per_box_masks

