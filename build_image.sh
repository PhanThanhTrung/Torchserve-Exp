#!/bin/bash

set -o errexit -o nounset -o pipefail

MACHINE=cpu
DOCKER_TAG="pytorch/torchserve:latest-cpu"
BASE_IMAGE="ubuntu:20.04"
UPDATE_BASE_IMAGE=false
CUDA_VERSION=""
BUILD_WITH_IPEX=false
PYTHON_VERSION=3.9

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-g, --gpu specify to use gpu"
          echo "-bi, --baseimage specify base docker image. Example: nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04 "
          echo "-cv, --cudaversion specify to cuda version to use"
          echo "-t, --tag specify tag name for docker image"
          echo "-ipex, --build-with-ipex specify to build with intel_extension_for_pytorch"
          echo "-py, --pythonversion specify to python version to use: Possible values: 3.8 3.9 3.10"
          exit 0
          ;;
        -g|--gpu)
          MACHINE=gpu
          DOCKER_TAG="pytorch/torchserve:latest-gpu"
          BASE_IMAGE="nvidia/cuda:12.1.1-base-ubuntu20.04"
          CUDA_VERSION="cu121"
          shift
          ;;
        -bi|--baseimage)
          BASE_IMAGE="$2"
          UPDATE_BASE_IMAGE=true
          shift
          shift
          ;;
        -lf|--use-local-serve-folder)
          USE_LOCAL_SERVE_FOLDER=true
          shift
          ;;
        -ipex|--build-with-ipex)
          BUILD_WITH_IPEX=true
          shift
          ;;
        -n|--nightly)
          BUILD_NIGHTLY=true
          shift
          ;;
        -py|--pythonversion)
          PYTHON_VERSION="$2"
          if [[ $PYTHON_VERSION = 3.8 || $PYTHON_VERSION = 3.9 || $PYTHON_VERSION = 3.10 || $PYTHON_VERSION = 3.11 ]]; then
            echo "Valid python version"
          else
            echo "Valid python versions are 3.8, 3.9 3.10 and 3.11"
            exit 1
          fi
          shift
          shift
          ;;
        -s|--source)
          BUILD_FROM_SRC=true
          shift
          ;;
        # With default ubuntu version 20.04
        -cv|--cudaversion)
          CUDA_VERSION="$2"
          if [ "${CUDA_VERSION}" == "cu121" ];
          then
            BASE_IMAGE="nvidia/cuda:12.1.0-base-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu118" ];
          then
            BASE_IMAGE="nvidia/cuda:11.8.0-base-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu117" ];
          then
            BASE_IMAGE="nvidia/cuda:11.7.1-base-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu116" ];
          then
            BASE_IMAGE="nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu113" ];
          then
            BASE_IMAGE="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
          elif [ "${CUDA_VERSION}" == "cu111" ];
          then
            BASE_IMAGE="nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04"
          else
            echo "CUDA version not supported"
            exit 1
          fi
          shift
          shift
          ;;
    esac
done

if [ "${MACHINE}" == "gpu" ] && $BUILD_WITH_IPEX ;
then
  echo "--gpu and --ipex are mutually exclusive. Please select one of them."
  exit 1
fi

if [[ $UPDATE_BASE_IMAGE == true && $MACHINE == "gpu" ]];
then
  echo "Incompatible options: -bi doesn't work with -g option"
  exit 1
fi

DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg USE_CUDA_VERSION="${CUDA_VERSION}"  --build-arg PYTHON_VERSION="${PYTHON_VERSION}"\
-t "${DOCKER_TAG}" --target production-image  ../
