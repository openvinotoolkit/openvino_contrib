FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt install -y software-properties-common libtbb-dev
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 net-tools wget
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | tee /etc/apt/sources.list.d/intel-gpu.list \
    && apt update

RUN apt-get update && apt-get install -y --no-install-recommends \
  clinfo hwinfo intel-igc-cm intel-level-zero-gpu intel-media-va-driver-non-free intel-opencl-icd ocl-icd-opencl-dev level-zero level-zero-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigc-dev libigdfcl-dev libigdgmm12 libigfxcmrt-dev libmfx1 libmfxgen1 libvpl2 libxatracker2 mesa-va-drivers mesa-vdpau-drivers mesa-vulkan-drivers udev va-driver-all libva2 libvpl-tools libmfx-dev vainfo ninja-build libgtk-3-dev libpng-dev libjpeg-dev libwebp-dev libtiff5-dev libopenexr-dev libopenblas-dev

RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.26.0/linux-npu-driver-v1.26.0.20251125-19665715237-ubuntu2404.tar.gz -O /tmp/linux-npu-driver-v1.26.0.20251125-19665715237-ubuntu2404.tar.gz

RUN mkdir -p /tmp/npu-driver \
    && tar -xvf /tmp/linux-npu-driver-v1.26.0.20251125-19665715237-ubuntu2404.tar.gz -C /tmp/npu-driver \
    && dpkg -i /tmp/npu-driver/*.deb
RUN rm -rf /tmp/*

RUN apt-get install -y ca-certificates git wget curl gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ollama_ov_server
ARG GOVERSION=1.24.1
RUN curl -fsSL https://golang.org/dl/go${GOVERSION}.linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH

ARG GENAI_NIGHTLY=2026.3.0.0.dev20260516
ARG GENAI_PKG=openvino_genai_ubuntu24_${GENAI_NIGHTLY}_x86_64
RUN wget https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/${GENAI_NIGHTLY}/${GENAI_PKG}.tar.gz
RUN tar -xzf ${GENAI_PKG}.tar.gz
ENV GENAI_DIR=/home/ollama_ov_server/${GENAI_PKG}

RUN source ${GENAI_DIR}/setupvars.sh

ENV CGO_ENABLED=1
ENV GODEBUG=cgocheck=0

ENV CGO_LDFLAGS=-L$GENAI_DIR/runtime/lib/intel64
ENV CGO_CFLAGS=-I$GENAI_DIR/runtime/include

WORKDIR /home/ollama_ov_server
RUN git clone https://github.com/zhaohb/ollama_openvino.git
WORKDIR /home/ollama_ov_server/ollama_openvino

RUN go build -o /usr/bin/ollama .

ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/bin/bash", "-c", "source ${GENAI_DIR}/setupvars.sh && /usr/bin/ollama serve"]