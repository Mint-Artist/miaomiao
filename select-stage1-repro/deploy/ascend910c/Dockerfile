ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/ascend910c/requirements-ascend.txt /tmp/requirements-ascend.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements-ascend.txt

COPY . /workspace

CMD ["bash"]
