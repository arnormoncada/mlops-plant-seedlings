# Base image
FROM python:3.11-slim AS base

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
# COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY src/session2/ src/session2/
COPY data/ data/

# Install requirements from scratch
# RUN pip install -r requirements.txt --no-cache-dir --verbose

# Install requirements from cache
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir

# Install project
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/session2/train.py"]

# to run train dockerfile
# docker build -f dockerfiles/train.dockerfile . -t train:latest