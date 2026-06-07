FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8710

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --no-compile -r requirements-prod.txt && \
    pip install --no-cache-dir --no-compile --no-deps "ultralytics>=8.3.0,<9" && \
    pip cache purge && \
    rm -rf /root/.cache/pip

COPY src /app/src
COPY weights/README.md /app/weights/README.md

RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    python -c "import sys; sys.path.insert(0, '/app'); from src.server import app; print('✓ queen bee detector import successful')"

EXPOSE 8710
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8710"]
