FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md app.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e .

# Prepare lightweight Middlebury & benchmark caches for instant live demo
RUN python scripts/download_datasets.py --dataset all --fast-middlebury

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
