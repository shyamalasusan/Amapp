

import os

from ultralytics import YOLO

DIOR_WEIGHTS = os.environ.get("AMAPP_DIOR_WEIGHTS", "dior_yolo.pt")
DOTA_WEIGHTS = os.environ.get("AMAPP_DOTA_WEIGHTS", "dota_obb_yolo.pt")

_model_dior = None
_model_dota = None


def get_model_dior():
    """Return the (lazily loaded, cached) DIOR-trained YOLO detector."""
    global _model_dior
    if _model_dior is None:
        _model_dior = YOLO(DIOR_WEIGHTS)
    return _model_dior


def get_model_dota():
    """Return the (lazily loaded, cached) DOTA-trained YOLO-OBB detector."""
    global _model_dota
    if _model_dota is None:
        _model_dota = YOLO(DOTA_WEIGHTS)
    return _model_dota
