#!/bin/bash

# set -o errexit -o nounset -o pipefail

MACHINE=cpu
DOCKER_TAG="pytorch/torchserve:latest-cpu"
BASE_IMAGE="ubuntu:20.04"
UPDATE_BASE_IMAGE=false
CUDA_VERSION=""
BUILD_WITH_IPEX=false
PYTHON_VERSION=3.10

export DOCKER_BUILDKIT=1
docker build --file Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg USE_CUDA_VERSION="${CUDA_VERSION}"  --build-arg PYTHON_VERSION="${PYTHON_VERSION}"\
-t "${DOCKER_TAG}" --target production-image  ../

docker build --file Dockerfile --build-arg BASE_IMAGE="ubuntu:20.04" --build-arg USE_CUDA_VERSION=""  --build-arg PYTHON_VERSION="3.10" -t "pytorch/torchserve:latest-cpu" --target production-image  ../