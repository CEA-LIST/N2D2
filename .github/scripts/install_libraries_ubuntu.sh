#!/bin/sh
################################################################################
#    (C) Copyright 2022 CEA LIST. All Rights Reserved.
#    Contributor(s): Vincent TEMPLIER (vincent.templier@cea.fr)
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


# Update libraries
sudo apt-get -y update


# Basic libraries for all N2D2 versions
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    gnuplot \
    libopencv-dev \
    python-dev \
    python3-dev \
    protobuf-compiler \
    libprotoc-dev \
    libjsoncpp-dev


# CUDA and CuDNN libraries for CUDA N2D2 versions
if [ -n "$USE_CUDA" ] ; then
    # Install the package for CUDA
    CUDA_PKG=cuda-11-0_11.0.3-1_amd64.deb
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/$CUDA_PKG
    sudo dpkg -i $CUDA_PKG
    rm $CUDA_PKG

    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

    # Install the package for CuDNN
    CUDNN_PKG = libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/$CUDNN_PKG
    sudo dpkg -i $CUDNN_PKG

    # Update the package lists
    sudo apt-get -y update

    # Install the CUDA and CuDNN packages
    CUDA_VERSION="11.0"

    sudo apt-get install -y --no-install-recommends cuda

    # Install the runtime library
    sudo apt-get install libcudnn8=8.0.5.39-1+cuda11.0

    # Install the developer library.
    sudo apt-get install libcudnn8-dev=8.0.5.39-1+cuda11.0

    # Manually create CUDA symlink
    ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
fi
