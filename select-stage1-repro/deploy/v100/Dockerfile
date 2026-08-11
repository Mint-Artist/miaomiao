FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY deploy/v100/requirements-v100.txt /tmp/requirements-v100.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements-v100.txt

COPY . /workspace

CMD ["/bin/bash"]
