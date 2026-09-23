FROM python:3.12-slim

# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJME_MODEL_PATH=/app/artifacts/model.joblib \
    PJME_HISTORY_PATH=/app/data/raw/PJME_hourly.csv \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir --no-deps .

COPY artifacts/model.joblib artifacts/model.json ./artifacts/
COPY data/raw/PJME_hourly.csv ./data/raw/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

# Platforms like Render/Railway inject $PORT; default 8000 locally.
CMD ["sh", "-c", "uvicorn pjme_forecast.service:app --host 0.0.0.0 --port ${PORT} --workers 1"]
