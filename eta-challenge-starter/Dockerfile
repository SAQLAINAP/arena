# ETA Challenge — inference-only image
# Build:  docker build -t eta-submission .
# Test:   docker run --rm -v $(pwd)/data:/work eta-submission /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

# libgomp1 required by LightGBM/XGBoost on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py features.py grade.py ./
COPY model.pkl ./

ENTRYPOINT ["python", "grade.py"]
