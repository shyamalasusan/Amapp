# AMAPP (linear, no-veto variant)

Adaptive Multi-Attribute Privacy Protection for aerial/satellite imagery.


## Layout

```
amapp/
├── config.py                 # DIOR/DOTA/sub-region classes + sensitivity tables
├── paths.py                   # dataset locations (EuroSAT / DIOR / DOTA)
├── models.py                  # lazy-loaded model\_dior / model\_dota (YOLO)
├── operators.py               # blur/pixelate/mask operators
├── scoring.py                 # whole-object MAPR risk formula
├── baselines.py               # global blur / fixed mask / single-attribute baselines
├── pipeline\_whole\_object.py   # whole-object AMAPP pipeline (bbox + OBB)
├── metrics.py                  # PSNR/SSIM/MAE + residual-detectability (bbox + OBB)
├── eval\_dota\_whole.py          # DOTA whole-object eval -> Tables 2/3
├── segmentation.py             # SAM box-prompted segmentation
├── part\_labeling.py            # part labeling for all 6 sub-region classes
├── subregion\_scoring.py        # per-part MAPR scoring + multi-region protection
├── pipeline\_subregion.py       # sub-region AMAPP pipeline (DIOR + DOTA/OBB)
├── eval\_subregion.py           # DIOR + DOTA sub-region eval -> Tables 2/3
├── eurosat\_eval.py             # EuroSAT downstream classifier utility test
├── calibration.py              # temperature scaling, ECE / D-ECE
├── ablation\_config.py          # Table 4 config-driven MAPR variants
├── ablation\_full.py            # Table 4: A0–A5 ablation + weight sensitivity + runtime
├── ablation\_subregion.py       # Table 4: sub-region ablation
└── figures.py                   # manuscript Figures 1–5

scripts/
└── setup\_dota.sh               # downloads + extracts DOTA v1.0
```

Each module's docstring says which `PART` letter(s) of the original script
it came from, so you can diff back against the source if needed.

## Setup

```bash
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements.txt
```

### Datasets

`amapp/paths.py` defaults to the Kaggle input mounts this project was
developed against:

* EuroSAT: `/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT`
* DIOR: `/kaggle/input/datasets/kefanzhan/diordataset/Dior/Dior`
* DOTA v1.0: fetched on demand (see below), not on Kaggle by default

Override with env vars if running elsewhere:

```bash
export AMAPP\_EUROSAT\_DIR=/path/to/EuroSAT
export AMAPP\_DIOR\_DIR=/path/to/Dior
export AMAPP\_DOTA\_DIR=/path/to/DOTAv1
```

To fetch DOTA v1.0:

```bash
bash scripts/setup\_dota.sh
# or manually:
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/DOTAv1.zip -P /tmp/data/
unzip -q /tmp/data/DOTAv1.zip -d /tmp/data/
```

### Detector weights

`amapp/models.py` lazily loads your trained detectors via `ultralytics.YOLO`:

```bash
export AMAPP\_DIOR\_WEIGHTS=/path/to/dior\_yolo.pt
export AMAPP\_DOTA\_WEIGHTS=/path/to/dota\_obb\_yolo.pt
```

Any eval/figure function that needs a detector calls `get\_model\_dior()` /
`get\_model\_dota()` internally, so the checkpoint is loaded once, on first
use.

You'll also need SAM weights (`sam\_b.pt`, loaded by `segmentation.py`) for
the sub-region pipeline.

## Usage sketch

```python
from amapp.pipeline\_whole\_object import run\_amapp\_linear
import cv2

image = cv2.imread("scene.jpg")
detections = \[...]  # (cls\_name, confidence, \[x1, y1, x2, y2]) from your detector
protected, scored = run\_amapp\_linear(image, detections, image.shape)
cv2.imwrite("protected.jpg", protected)
```

Running a full DOTA whole-object evaluation:

```python
from amapp.eval\_dota\_whole import run\_dota\_whole\_object\_linear\_eval
from amapp.paths import DOTA\_DIR

table2, table3 = run\_dota\_whole\_object\_linear\_eval(f"{DOTA\_DIR}/images/val")
```

## Output paths

Several eval/figure functions hard-code Kaggle-style output paths
(`/kaggle/working/...`) for saving result CSVs and figures. If you're not
running on Kaggle, update these to a local `outputs/` directory (or route
through `amapp.paths.OUTPUT\_DIR`) before running full evaluations.

