FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        gnuplot \
        libopencv-dev \
        libcv-dev \
        libhighgui-dev

ENV N2D2_ROOT=/opt/N2D2
WORKDIR $N2D2_ROOT

RUN git clone https://github.com/CEA-LIST/N2D2.git . && \
    mkdir build && cd build && \
    cmake .. && \
    make -j"$(nproc)"

ENV N2D2_MODELS $N2D2_ROOT/models
ENV PATH $N2D2_ROOT/build/bin/exec:$PATH

WORKDIR /workspace

