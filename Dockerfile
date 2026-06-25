FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt /workspace/requirements-lock.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements-lock.txt

COPY . /workspace

CMD ["python3", "-m", "pytest", "-q"]
