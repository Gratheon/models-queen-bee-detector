from pathlib import Path
from typing import Any
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

WEIGHTS_PATH = Path(os.getenv("QUEEN_BEE_DETECTOR_WEIGHTS", "weights/best.pt"))
MAX_UPLOAD_BYTES = int(os.getenv("QUEEN_BEE_DETECTOR_MAX_UPLOAD_BYTES", "8000000"))
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.45

app = FastAPI(title="Gratheon Queen Bee Detector", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Model weights not found at {WEIGHTS_PATH}")
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "weights_present": WEIGHTS_PATH.exists()}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = DEFAULT_CONFIDENCE, iou: float = DEFAULT_IOU) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Uploaded file is larger than {MAX_UPLOAD_BYTES} bytes")
    if not 0 < conf <= 1:
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1")
    if not 0 < iou <= 1:
        raise HTTPException(status_code=400, detail="iou must be between 0 and 1")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a decodable image")

    model = get_model()
    predictions = model.predict(image, conf=conf, iou=iou, verbose=False)

    detections: list[dict[str, Any]] = []
    names = model.names
    for result in predictions:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, class_id)),
                    "confidence": float(box.conf[0].item()),
                    "box": [float(v) for v in box.xyxy[0].tolist()],
                }
            )

    if not detections:
        return {"message": "Nothing found", "result": []}
    return {"message": "File processed successfully", "result": detections}
