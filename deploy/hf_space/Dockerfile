# Hugging Face Docker Space: serves pre-trained artifacts (no Redis on the
# free tier — the engine falls back to the in-process feature store).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt faiss-cpu

COPY src/ src/
COPY artifacts/ artifacts/

ENV PYTHONPATH=/app/src \
    ARTIFACTS_DIR=/app/artifacts \
    HOME=/tmp

EXPOSE 7860
CMD ["uvicorn", "recsys.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
