

import cv2
import numpy as np


def _pixelate_region(region, block_size=12):
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return region
    small = cv2.resize(
        region,
        (max(1, w // block_size), max(1, h // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_operator(image, bbox, operator):
    """Apply `operator` ("none"/"blur"/"pixelate"/"mask") to an axis-aligned
    bbox region [x1, y1, x2, y2] of `image`. Returns a new image."""
    if operator == "none":
        return image

    protected = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return protected

    region = protected[y1:y2, x1:x2]

    if operator == "blur":
        k = max(3, (min(region.shape[:2]) // 4) | 1)  # odd kernel, scales with region
        region_out = cv2.GaussianBlur(region, (k, k), 0)
    elif operator == "pixelate":
        region_out = _pixelate_region(region)
    elif operator == "mask":
        region_out = np.zeros_like(region)
    else:
        raise ValueError(f"Unknown operator: {operator}")

    protected[y1:y2, x1:x2] = region_out
    return protected


def apply_operator_obb(image, corners, operator):
    """Apply `operator` to an oriented region defined by `corners`
    (a flat/nested list of 4 (x, y) polygon points). Returns a new image."""
    if operator == "none":
        return image

    protected = image.copy()
    pts = np.array(corners, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    aabb = [x1, y1, x2, y2]

    # Build a polygon mask so only the oriented region (not its full AABB)
    # is modified.
    poly_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts.astype(np.int32)], 1)
    poly_mask = poly_mask.astype(bool)

    fully_processed = apply_operator(protected, aabb, operator)
    protected[poly_mask] = fully_processed[poly_mask]
    return protected
