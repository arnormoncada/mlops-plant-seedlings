FROM python:3.10-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install wandb
COPY wandb_tester.py wandb_tester.py
ENTRYPOINT ["python", "-u", "wandb_tester.py"]