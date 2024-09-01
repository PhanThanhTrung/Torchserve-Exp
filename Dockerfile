ARG BASE_IMAGE=ubuntu:rolling
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE}
ENV PYTHONUNBUFFERED TRUE
ARG USE_CUDA_VERSION=""

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip  python3-pip && \
    apt-get install --no-install-recommends -y \
        ca-certificates \
        g++ \
        python3-distutils \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
        openjdk-17-jdk \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python$PYTHON_VERSION -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"
RUN python -m pip install -U pip setuptools
RUN export USE_CUDA=1

COPY ./ /root
WORKDIR /root/

RUN \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        if [ "$USE_CUDA_VERSION" ]; then \
            python3 install_dependencies.py --cuda $USE_CUDA_VERSION;\
        else \
            python3 install_dependencies.py;\
        fi; \
    else \
        python3 install_dependencies.py; \
    fi

RUN pip3 install --no-cache-dir torchserve torch-model-archiver torch-workflow-archiver
RUN chmod +x dockerd-entrypoint.sh

ENTRYPOINT ["/bin/bash", "./dockerd-entrypoint.sh"]
CMD ["serve"]