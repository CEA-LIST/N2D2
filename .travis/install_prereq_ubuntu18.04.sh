#!/bin/sh
################################################################################
#    (C) Copyright 2016 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
#
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
################################################################################

USE_CUDA=1

apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    gnuplot \
    libopencv-dev \
    libpugixml-dev \
    mongodb-dev \
    libjsoncpp-dev \
    libprotobuf-dev \
    protobuf-compiler

if [ -n "$USE_CUDA" ] ; then
    # Install the "repo" package for CUDA
    CUDA_REPO_PKG=cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/$CUDA_REPO_PKG
    dpkg -i $CUDA_REPO_PKG
    rm $CUDA_REPO_PKG

    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

    # Install the "repo" package for CuDNN
    ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/$ML_REPO_PKG
    dpkg -i $ML_REPO_PKG

    # Update the package lists
    apt-get -y update

    # Install the CUDA and CuDNN packages
    CUDA_PKG_VERSION="10-0"
    CUDA_VERSION="10.0"

    apt-get install -y --no-install-recommends \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        libcudnn7-dev

    # Manually create CUDA symlink
    ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
fi
