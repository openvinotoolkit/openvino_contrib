FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt install -y software-properties-common libtbb-dev
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 net-tools
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN apt-get install -y ca-certificates git wget curl gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ollama_ov_server
ARG GOVERSION=1.24.1
RUN curl -fsSL https://golang.org/dl/go${GOVERSION}.linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH

RUN wget https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2025.2.0.0.dev20250513/openvino_genai_ubuntu24_2025.2.0.0.dev20250513_x86_64.tar.gz
RUN tar -xzf openvino_genai_ubuntu24_2025.2.0.0.dev20250513_x86_64.tar.gz
ENV GENAI_DIR=/home/ollama_ov_server/openvino_genai_ubuntu24_2025.2.0.0.dev20250513_x86_64

RUN source /home/ollama_ov_server/openvino_genai_ubuntu24_2025.2.0.0.dev20250513_x86_64/setupvars.sh

ENV CGO_ENABLED=1
ENV GODEBUG=cgocheck=0

ENV CGO_LDFLAGS=-L$GENAI_DIR/runtime/lib/intel64
ENV CGO_CFLAGS=-I$GENAI_DIR/runtime/include

WORKDIR /home/ollama_ov_server
RUN git clone https://github.com/openvinotoolkit/openvino_contrib.git
WORKDIR /home/ollama_ov_server/openvino_contrib/modules/ollama_openvino

RUN go build -o /usr/bin/ollama .

ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/bin/bash", "-c", "source /home/ollama_ov_server/openvino_genai_ubuntu24_2025.2.0.0.dev20250513_x86_64/setupvars.sh && /usr/bin/ollama serve"]