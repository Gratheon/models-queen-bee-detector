import asyncio
from pathlib import Path

from src import server


def test_readme_documents_browser_mode():
    text = Path("README.md").read_text()
    assert "/warehouse/queens/detect" in text
    assert "VITE_QUEEN_BEE_DETECTOR_URL" in text


def test_dataset_catalog_has_primary_dataset():
    text = Path("datasets/catalog.yaml").read_text()
    assert "honey-bee-detection-model-zgjnb" in text
    assert "queen" in text.lower()


def test_health_endpoint_reports_weight_status():
    body = server.health()

    assert body["status"] == "ok"
    assert "weights_present" in body
    assert "weights_path" in body


def test_root_upload_form_matches_bee_detector_contract():
    html = server.upload_form()

    assert 'method="POST"' in html
    assert 'name="file"' in html


def test_root_and_detect_post_routes_share_file_contract():
    post_routes = {
        route.path
        for route in server.app.routes
        if hasattr(route, "methods") and "POST" in route.methods
    }

    assert "/" in post_routes
    assert "/detect" in post_routes


def test_detect_endpoint_keeps_existing_contract(monkeypatch):
    class FakeUploadFile:
        filename = "queen.jpg"

        async def read(self):
            return b"fake-image"

    async def fake_run_detection(file, conf, iou):
        return {
            "message": "File processed successfully",
            "result": [
                {"class_id": 0, "class_name": "queen", "confidence": 0.91, "box": [10, 20, 110, 180]}
            ],
        }

    monkeypatch.setattr(server, "run_detection", fake_run_detection)

    response = asyncio.run(server.detect(file=FakeUploadFile()))

    assert response["message"] == "File processed successfully"
    assert response["result"][0]["class_name"] == "queen"
