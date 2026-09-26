
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# PART F — EVALUATION METRICS (utility, residual detectability)
# =============================================================================

def compute_utility_metrics(original, protected):
    mae = np.mean(np.abs(original.astype(float) - protected.astype(float)))
    psnr_val = psnr(original, protected, data_range=255)
    ssim_val = ssim(original, protected, channel_axis=2, data_range=255)
    return mae, psnr_val, ssim_val

def compute_residual_detectability(protected_image, model, class_names, original_count):
    """Axis-aligned-box counterpart of compute_residual_detectability_obb:
    re-detects on the protected image and returns the fraction of the
    original detection count that still shows up."""
    tmp_path = "/tmp/_tmp_protected.jpg"
    cv2.imwrite(tmp_path, protected_image)
    results = model.predict(source=tmp_path, save=False, conf=0.4, verbose=False)
    residual_count = 0
    for r in results:
        if r.boxes is not None:
            residual_count += len(r.boxes)
    return residual_count / original_count if original_count > 0 else 0.0


def compute_residual_detectability_obb(protected_image, model, original_count):
    tmp_path = "/tmp/_tmp_protected_dota.jpg"
    cv2.imwrite(tmp_path, protected_image)
    results = model.predict(source=tmp_path, save=False, conf=0.4, verbose=False)
    residual_count = 0
    for r in results:
        if r.obb is not None:
            residual_count += len(r.obb)
    return residual_count / original_count if original_count > 0 else 0.0

