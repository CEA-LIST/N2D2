FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        gnuplot \
        libopencv-dev \
        python-dev \
        python3-dev \
        protobuf-compiler \
        libprotoc-dev

ENV N2D2_ROOT=/opt/N2D2
WORKDIR $N2D2_ROOT

RUN git clone --recursive https://github.com/CEA-LIST/N2D2.git . && \
    mkdir build && cd build && \
    cmake .. && \
    make -j"$(nproc)"

ENV N2D2_MODELS $N2D2_ROOT/models
ENV PATH $N2D2_ROOT/build/bin/exec:$PATH

WORKDIR /workspace

