FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
ARG enable_tensorrt
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# nv-tensorrt-repo-*.deb packages
COPY *.deb  /

SHELL ["/bin/bash", "-c"]
RUN set -exuo pipefail ; \
	rm -f /etc/apt/apt.conf.d/docker-clean ; \
	echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache ; \
	apt update ; \
	apt install -y \
		gnupg2 \
		software-properties-common \
	; \
	if [[ ${enable_tensorrt-} == "1" ]] ; then dpkg -i /nv-tensorrt-repo-*.deb ; fi; \
	apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub ; \
	add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" ; \
	apt update ; \
	DEBIAN_FRONTEND=noninteractive apt install -y \
		./libcudnn*.deb \
		autoconf \
		automake \
		build-essential \
		ca-certificates \
		clang-format-9 \
		cuda-runtime-11-8 \
		cuda-11-8 \
		curl \
		g++-multilib \
		gcc-multilib \
		git \
		git-lfs \
		gstreamer1.0-plugins-base \
		libavcodec-dev \
		libavformat-dev \
		libboost-regex-dev \
		libcairo2-dev \
		libcutensor1=1.6.1.5-1 \
		libcutensor-dev=1.6.1.5-1 \
		libglib2.0-dev \
		libgstreamer1.0-0 \
		libgtk2.0-dev \
		libopenblas-dev \
		libpango1.0-dev \
		libpng-dev \
		libssl-dev \
		libswscale-dev \
		libtool \
		libusb-1.0-0-dev \
		libzstd-dev \
		pkg-config \
		python \
		python3-pip \
		shellcheck \
		time \
		unzip \
		wget \
		cuda-drivers=520.61.05-1\
	; \
	pip3 install cython ; \
	curl -L https://github.com/Kitware/ninja/releases/download/v1.10.0.gfb670.kitware.jobserver-1/ninja-1.10.0.gfb670.kitware.jobserver-1_x86_64-linux-gnu.tar.gz | tar xzv --strip-components=1 -C /usr/local/bin ; \
	curl https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz -L | tar xvzC /usr/local/bin --strip-components=1 --wildcards '*/sccache' ; \
	chmod a+x /usr/local/bin/sccache ; \
	curl https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.tar.gz -L | tar xzvC /usr/local --exclude={doc,man} --strip-components=1 ; \
	curl -L https://github.com/ccache/ccache/releases/download/v4.3/ccache-4.3.tar.xz | tar xJv ; \
	cd ccache-4.3 ; \
	mkdir build ; \
	cd build ; \
	cmake -DCMAKE_BUILD_TYPE=Release -G Ninja .. ; \
	ninja -v install ; \
	cd / ; \
	rm -r *.deb ccache*

ENV CUDA_PATH=/usr/local/cuda
ENV PATH=${CUDA_PATH}/bin:$PATH
