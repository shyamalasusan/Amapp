

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .models import get_model_dior

# PART P — EUROSAT DOWNSTREAM UTILITY TEST (CPU)
# =============================================================================

DEVICE = torch.device("cpu")

CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

def load_eurosat_classifier(checkpoint_path):
    model_cls = models.resnet18(weights=None)
    model_cls.fc = nn.Linear(model_cls.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model_cls.load_state_dict(state_dict)
    model_cls = model_cls.to(DEVICE)
    model_cls.eval()
    return model_cls

eurosat_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image_array(image_bgr, model_cls, transform):
    """Takes a cv2 (BGR, numpy) image, returns predicted class index. CPU only."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model_cls(tensor)
        _, pred = torch.max(output, 1)
    return int(pred.item())

def run_eurosat_downstream_test(base_dir, model_cls, sample_size=293, seed=42, conf_threshold=0.4):
    """
    base_dir: EuroSAT root folder containing validation.csv and images
              (same layout as used during EuroSAT classifier training).
    model_cls: loaded via load_eurosat_classifier().
    """
    val_df = pd.read_csv(os.path.join(base_dir, "validation.csv"))
    val_sample = val_df.sample(n=min(sample_size, len(val_df)), random_state=seed)

    correct_original, correct_protected, total = 0, 0, 0
    detections_found_count = 0

    for idx, row in val_sample.iterrows():
        img_path = os.path.join(base_dir, row['Filename'])
        if not os.path.exists(img_path):
            # Handle validation.csv formats where Filename already includes
            # the full relative path from base_dir's parent, or an absolute
            # path -- fall back to using row['Filename'] directly rather
            # than silently skipping every image on a path mismatch.
            if os.path.exists(row['Filename']):
                img_path = row['Filename']
        true_label = int(row['Label'])

        image = cv2.imread(img_path)
        if image is None:
            continue
        img_shape = image.shape

        results = get_model_dior().predict(source=img_path, save=False, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = DIOR_CLASSES[cls_id]
                confidence = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                detections.append((cls_name, confidence, xyxy))

        if detections:
            detections_found_count += 1
            protected_image, _ = run_amapp_linear(image, detections, img_shape)
        else:
            protected_image = image.copy()

        pred_original = classify_image_array(image, model_cls, eurosat_transform)
        pred_protected = classify_image_array(protected_image, model_cls, eurosat_transform)

        if pred_original == true_label:
            correct_original += 1
        if pred_protected == true_label:
            correct_protected += 1
        total += 1

        if total % 25 == 0:
            print(f"Processed {total}/{len(val_sample)} images...")

    if total == 0:
        print(f"\nERROR: 0 of {len(val_sample)} images could be loaded from '{base_dir}'. "
              f"Check that base_dir and validation.csv's Filename column are consistent "
              f"(e.g. row['Filename'] may need to be joined differently with base_dir).")
        return {"total": 0, "detections_found_count": 0, "accuracy_original": None, "accuracy_protected": None}

    print(f"\nEvaluated {total} EuroSAT images")
    print(f"Images with at least 1 DIOR-model detection: {detections_found_count} "
          f"({100*detections_found_count/total:.1f}%)")
    print(f"Classification accuracy, ORIGINAL images: {100*correct_original/total:.2f}%")
    print(f"Classification accuracy, AMAPP (linear)-PROTECTED images: {100*correct_protected/total:.2f}%")
    print(f"Accuracy delta: {100*(correct_protected-correct_original)/total:+.2f} percentage points")

    return {
        "total": total,
        "detections_found_count": detections_found_count,
        "accuracy_original": 100 * correct_original / total,
        "accuracy_protected": 100 * correct_protected / total,
    }


