# gratheon/models-queen-bee-detector

Queen bee object detector for Gratheon.

This repo mirrors the purpose of `models-bee-detector`, but focuses on detecting queen bees among worker bees, drones, pollen bees, and other frame/background content.

## Current status

- Training, evaluation, export, and HTTP inference pipeline are prepared.
- Baseline model trained with `yolov8n.pt`, `imgsz=512`, `epochs=60` on merged queen datasets.
- Test metrics for `weights/best.pt`: precision `0.9727`, recall `0.8590`, mAP50 `0.9187`, mAP50-95 `0.6114`.
- Weights, datasets, runs, `.env`, and browser exports are intentionally not committed.

## Dataset candidates

Best candidates found online:

| Priority | Dataset | URL | Classes / size | Notes |
|---|---|---|---|---|
| 1 | Honey Bee Detection Model by Matt Nudi | https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb/dataset/2 | 909 images; workers/drones/queens/pollenbees | Best starting dataset: includes non-queen bee classes, which is important to reduce queen false positives. Existing `models-bee-detector` already credits this source. |
| 2 | Queen_Bee by SharedWS | https://universe.roboflow.com/sharedws/queen_bee-kcfnv | 583 `queenBee` images | Good queen-focused augmentation dataset; normalize class names to `queen`. |
| 3 | Bee Detection by Workspace 1 | https://universe.roboflow.com/workspace-1-tu7e5/bee-detection-h59jf | `bee`, `drone`, `pollenbee`, `queen`; CC BY 4.0 | Useful because license is explicit. |
| 4 | Queen Bee by Detection | https://universe.roboflow.com/detection-a6lof/queen-bee | 97 `Queen-Bee` images; CC BY 4.0 | Too small alone, useful supplement. |
| 5 | bees by Andrew Hofer | https://universe.roboflow.com/andrew-hofer-1qh7e/bees-ytrmp | 6423 `bee`, `queen`, `mite` images per search snippet | Promising, verify license before use. |

See `datasets/catalog.yaml` for machine-readable notes.

## Setup

```bash
cd models-queen-bee-detector
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Roboflow dataset

```bash
export ROBOFLOW_API_KEY=...
python scripts/download_dataset.py \
  --workspace matt-nudi \
  --project honey-bee-detection-model-zgjnb \
  --version 2 \
  --format yolov8 \
  --output datasets/queen-bee
```

The script writes the resolved dataset YAML path to stdout. Roboflow may create a nested folder; use the printed YAML in training.
## Prepare merged queen dataset

Download the Roboflow datasets listed in `datasets/catalog.yaml`, then merge them into a single-class `queen` dataset:

```bash
python scripts/merge_queen_dataset.py \
  --output datasets/queen-bee-merged \
  datasets/queen-bee-primary \
  datasets/bee-detection-workspace1 \
  datasets/queen-bee-detection-v2 \
  datasets/healthy-queen-bee \
  datasets/queen-bee-detection-vincent \
  datasets/bee-queen-scaner \
  datasets/queen-arpan \
  datasets/queen-detector \
  datasets/queen-local
```

The current merged dataset summary was:

```txt
train: 18877 images, 12764 queen boxes
valid: 2847 images, 1838 queen boxes
test: 1416 images, 914 queen boxes
```

`merge_queen_dataset.py` normalizes all queen-like class names to class `0 queen`, keeps non-queen images as negative/background samples, and converts segmentation polygon labels to bounding boxes.

## Train

```bash
python scripts/train.py \
  --data datasets/queen-bee-merged/data.yaml \
  --model yolov8n.pt \
  --epochs 60 \
  --imgsz 512 \
  --batch 2 \
  --device mps \
  --project runs/queen-bee-detector \
  --name merged-queen-yolov8n-img512
```

For higher accuracy, try `yolov8s.pt`, `imgsz=640`, and a larger batch on a machine with more GPU memory.

## Evaluate

```bash
python scripts/evaluate.py \
  --weights weights/best.pt \
  --data datasets/queen-bee-merged/data.yaml \
  --imgsz 512 \
  --split test \
  --project runs/queen-bee-detector \
  --name test-best
```

Expected baseline test metrics:

```txt
precision: 0.9727
recall: 0.8590
mAP50: 0.9187
mAP50-95: 0.6114
```

## Export for browser

Recommended browser path is ONNX + `onnxruntime-web` in web-app:

```bash
python scripts/export_model.py \
  --weights weights/best.pt \
  --format onnx \
  --imgsz 512
```

You can also try TensorFlow.js export if your environment supports the required dependencies:

```bash
python scripts/export_model.py --weights weights/best.pt --format tfjs --imgsz 512
```

## HTTP inference service

```bash
# Copy trained weights into weights/best.pt first if they are not already there.
uvicorn src.server:app --host 0.0.0.0 --port 8710
```

Optional runtime settings:

```bash
export QUEEN_BEE_DETECTOR_WEIGHTS=weights/best.pt
export QUEEN_BEE_DETECTOR_MAX_UPLOAD_BYTES=8000000
export QUEEN_BEE_DETECTOR_CONF=0.3
export QUEEN_BEE_DETECTOR_IOU=0.45
export OTEL_SERVICE_NAME=models-queen-bee-detector
```

The service uses the shared Gratheon Python logger via:

```txt
gratheon-log-lib @ https://github.com/Gratheon/log-lib-py/archive/03b30ba.zip
```

It emits request-scoped logs with `request_id`. To export logs to an OTLP collector, set `OTEL_EXPORTER_OTLP_ENDPOINT`.

Or run with Docker:

```bash
docker compose up --build
```

Endpoints:

- `GET /health` — health check with model weight presence.
- `GET /` — simple upload form, matching `models-bee-detector` behavior.
- `POST /` and `POST /detect` — multipart upload with `file` field. Optional query params: `conf`, `iou`.

POST an image:

```bash
curl -F file=@queen.jpg http://localhost:8710/detect
curl -F file=@queen.jpg http://localhost:8710/
```

Response shape:

```json
{
  "message": "File processed successfully",
  "result": [
    {
      "class_id": 0,
      "class_name": "queen",
      "confidence": 0.91,
      "box": [10, 20, 110, 180]
    }
  ]
}
```

## Browser/web-app mode

`web-app` now has a camera page at `/warehouse/queens/detect` that:

1. Opens browser camera via `navigator.mediaDevices.getUserMedia`.
2. Captures frames periodically to a canvas.
3. Sends JPEG frames to the queen detector HTTP service.
4. Draws detection boxes over the live video.

Configuration:

```bash
VITE_QUEEN_BEE_DETECTOR_URL=http://localhost:8710/detect pnpm dev
```

This is a server-backed browser mode. Full offline in-browser inference needs the exported ONNX/TFJS artifact to be copied into `web-app/public/models/queen-bee-detector/` and a small frontend inference wrapper added around `onnxruntime-web` or tfjs.

## License notes

- Dataset licenses differ. Verify each Roboflow project license before mixing datasets or shipping a commercial model.
- Ultralytics YOLO has licensing constraints. Confirm current Ultralytics license terms for your intended deployment.
