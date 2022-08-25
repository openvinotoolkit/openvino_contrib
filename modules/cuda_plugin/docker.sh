#!/usr/bin/env bash

set -xeuo pipefail

[[ "$1" == "install" || "$1" == "build" || "$1" == "run" ]] || {
  printf "One of the following command should be provided:\n\
  - install (allow to install docker with nvidia-toolkit support)\n\
  - build (allow build docker container from Dockerfile)\n\
  - run (allow run some command in docker container)\n";
  exit 1;
}

function install() {
    sudo apt-get update
    sudo apt-get install -y curl gnupg2

    . /etc/os-release
    distribution=$ID$VERSION_ID
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo apt-get install -y docker.io

    sudo usermod -aG docker $USER
    sudo systemctl restart docker
}

function build() {
    if [[ ! ./Dockerfile -ef ${CUDA_PACKAGES_PATH}/Dockerfile ]]; then
        cp Dockerfile "${CUDA_PACKAGES_PATH}"/
    fi

    pushd "${CUDA_PACKAGES_PATH}"
    docker build -t openvino/cudaplugin-2022.1 .
    if [[ ! ./Dockerfile -ef ${CUDA_PACKAGES_PATH}/Dockerfile ]]; then
        rm -f "${CUDA_PACKAGES_PATH}"/Dockerfile
    fi
    popd
}

function run() {
    OPENVINO_TEMP_PATH=$OPENVINO_HOME/inference-engine/temp
    OPENCV_PATH=$OPENVINO_TEMP_PATH/opencv_4.5.1_ubuntu18/opencv
    TBB_PATH=$OPENVINO_TEMP_PATH/tbb
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+LD_LIBRARY_PATH:}${TBB_PATH}/lib:${OPENCV_PATH}/lib:${OPENVINO_HOME}/bin/intel64/${BUILD_TYPE}/lib"

    cd "${OPENVINO_HOME}/bin/intel64/${BUILD_TYPE}"
    docker run --gpus all -e LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
                          -v "${PWD}:${PWD}" \
                          -v "${OPENVINO_TEMP_PATH}:${OPENVINO_TEMP_PATH}" \
                          -v "${USER_SHARE_PATH}:${USER_SHARE_PATH}" \
                          -w "${PWD}" openvino/cudaplugin-2022.1 \
                          ${USER_APP}
}

if [[ "$1" == "install" ]]; then
  install
elif [[ "$1" == "build" ]]; then
  [[ -n ${CUDA_PACKAGES_PATH} ]] || { echo "CUDA_PACKAGES_PATH environment variable is expected"; exit 1; }
  build
elif [[ "$1" == "run" ]]; then
  BUILD_TYPE=${BUILD_TYPE:-Release}

  [[ -n ${OPENVINO_HOME} ]] || { echo "OPENVINO_HOME environment variable is expected"; exit 1; }

  [[ -n $2 ]] || { echo "USER_SHARE_PATH as second argument is expected"; exit 1; }
  [[ -n $3 ]] || { echo "USER_APP as third argument is expected"; exit 1; }

  USER_SHARE_PATH=$2
  USER_APP=$3

  run
fi
