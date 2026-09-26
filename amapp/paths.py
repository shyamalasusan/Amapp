
import os

EUROSAT_DIR = os.environ.get(
    "AMAPP_EUROSAT_DIR",
    "/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT",
)

DIOR_DIR = os.environ.get(
    "AMAPP_DIOR_DIR",
    "/kaggle/input/datasets/kefanzhan/diordataset/Dior/Dior",
)

# DOTA v1.0 isn't on Kaggle by default in this project; it's fetched with:
#   wget https://github.com/ultralytics/yolov5/releases/download/v1.0/DOTAv1.zip -P /tmp/data/
#   unzip -q /tmp/data/DOTAv1.zip -d /tmp/data/
DOTA_DIR = os.environ.get("AMAPP_DOTA_DIR", "/tmp/data/DOTAv1")

OUTPUT_DIR = os.environ.get("AMAPP_OUTPUT_DIR", "/kaggle/working")
