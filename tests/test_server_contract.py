from pathlib import Path


def test_readme_documents_browser_mode():
    text = Path("README.md").read_text()
    assert "/warehouse/queens/detect" in text
    assert "VITE_QUEEN_BEE_DETECTOR_URL" in text


def test_dataset_catalog_has_primary_dataset():
    text = Path("datasets/catalog.yaml").read_text()
    assert "honey-bee-detection-model-zgjnb" in text
    assert "queen" in text.lower()
