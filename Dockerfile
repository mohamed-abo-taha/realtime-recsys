# Online path only: serve pre-trained artifacts. Training happens outside
# the container (offline path), so the image ships CPU-only torch and stays lean.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt
# ANN backend: ~40MB wheel, turns the index from exact numpy (~5ms) into HNSW (~0.1ms)
RUN pip install --no-cache-dir faiss-cpu

COPY src/ src/
COPY artifacts/ artifacts/

ENV PYTHONPATH=/app/src \
    ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "recsys.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
