# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY models models/
COPY models/classes.json models/classes.json

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["sh", "-c", "uvicorn src.plant_seedlings.api:app --host 0.0.0.0 --port $PORT"]
# docker build -f dockerfiles/api.dockerfile . -t seedlings_app:latest
# docker run -p 8000:8000 -e PORT=8000 seedlings_app:latest
