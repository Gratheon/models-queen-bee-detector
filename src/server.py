from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import os
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from gratheon_log_lib import bind_context, clear_context, configure, error_enriched, info, warn
from ultralytics import YOLO

WEIGHTS_PATH = Path(os.getenv("QUEEN_BEE_DETECTOR_WEIGHTS", "weights/best.pt"))
MAX_UPLOAD_BYTES = int(os.getenv("QUEEN_BEE_DETECTOR_MAX_UPLOAD_BYTES", "8000000"))
DEFAULT_CONFIDENCE = float(os.getenv("QUEEN_BEE_DETECTOR_CONF", "0.3"))
DEFAULT_IOU = float(os.getenv("QUEEN_BEE_DETECTOR_IOU", "0.45"))
SERVICE_PORT = int(os.getenv("PORT", "8710"))

configure()

@asynccontextmanager
async def lifespan(app: FastAPI):
    info(
        "starting queen bee detector server",
        {
            "port": SERVICE_PORT,
            "weights": str(WEIGHTS_PATH),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "default_conf": DEFAULT_CONFIDENCE,
            "default_iou": DEFAULT_IOU,
        },
    )
    yield


app = FastAPI(title="Gratheon Queen Bee Detector", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
_model: YOLO | None = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    bind_context(request_id=request_id)

    info(
        "incoming queen bee detector request",
        {
            "path": request.url.path,
            "method": request.method,
            "remote_addr": request.client.host if request.client else None,
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        },
    )

    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        info(
            "queen bee detector request finished",
            {
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["x-request-id"] = request_id
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        error_enriched(
            "queen bee detector request failed",
            exc,
            {
                "path": request.url.path,
                "method": request.method,
                "duration_ms": duration_ms,
            },
        )
        raise
    finally:
        clear_context()


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            warn("queen bee detector weights missing", {"weights": str(WEIGHTS_PATH)})
            raise HTTPException(status_code=503, detail=f"Model weights not found at {WEIGHTS_PATH}")
        info("loading queen bee detector model", {"weights": str(WEIGHTS_PATH)})
        _model = YOLO(str(WEIGHTS_PATH))
        info("queen bee detector model loaded", {"classes": _model.names})
    return _model


@app.get("/", response_class=HTMLResponse)
def upload_form() -> str:
    return """
    <html>
    <body>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" />
        <input type="submit" value="Upload" />
    </form>
    </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "weights_present": WEIGHTS_PATH.exists(),
        "weights_path": str(WEIGHTS_PATH),
    }


async def run_detection(file: UploadFile, conf: float, iou: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    image_bytes = await file.read()
    if not image_bytes:
        warn("rejecting queen bee detector request, empty upload", {"filename": file.filename})
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        warn(
            "rejecting queen bee detector request, upload too large",
            {"filename": file.filename, "image_bytes": len(image_bytes), "max_upload_bytes": MAX_UPLOAD_BYTES},
        )
        raise HTTPException(status_code=413, detail=f"Uploaded file is larger than {MAX_UPLOAD_BYTES} bytes")
    if not 0 < conf <= 1:
        warn("rejecting queen bee detector request, invalid confidence", {"conf": conf})
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1")
    if not 0 < iou <= 1:
        warn("rejecting queen bee detector request, invalid iou", {"iou": iou})
        raise HTTPException(status_code=400, detail="iou must be between 0 and 1")

    info(
        "uploaded queen bee image received",
        {"filename": file.filename, "image_bytes": len(image_bytes), "conf": conf, "iou": iou},
    )

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        warn("rejecting queen bee detector request, undecodable image", {"filename": file.filename})
        raise HTTPException(status_code=400, detail="Uploaded file is not a decodable image")

    model = get_model()
    info(
        "starting queen bee detection inference",
        {
            "weights": str(WEIGHTS_PATH),
            "image_shape": list(image.shape),
            "conf": conf,
            "iou": iou,
        },
    )
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

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    info(
        "queen bee detector request processed",
        {"detections": len(detections), "duration_ms": duration_ms},
    )

    if not detections:
        return {"message": "Nothing found", "result": []}
    return {"message": "File processed successfully", "result": detections}


@app.post("/")
@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = DEFAULT_CONFIDENCE, iou: float = DEFAULT_IOU) -> dict[str, Any]:
    return await run_detection(file=file, conf=conf, iou=iou)
