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
    git \
    wget \
    gnuplot \
    libopencv-dev \
    python-dev \
    python3-dev \
    protobuf-compiler \
    libprotoc-dev \
    libjsoncpp-dev


# Installation of CMake
if [ -n "$USE_CUDA" ] ; then

    # It is not possible to build with the APT version of CMake a CUDA project 
    # with a CUDA version higher than 11 on a device which doesn't possess a GPU.
    # It is required to have a newer version of CMake
    # Ref: https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
    sudo apt remove -y --purge --auto-remove cmake
    sudo apt update -y
    sudo apt install -y software-properties-common lsb-release
    sudo apt clean all -y
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt update -y
    sudo apt install -y kitware-archive-keyring
    rm /etc/apt/trusted.gpg.d/kitware.gpg
    sudo apt update -y
    sudo apt install -y cmake

else
    # The basic version proposed by the APT deposit 
    # is enough for non-CUDA versions
    sudo apt-get install -y cmake
fi


# CUDA and CuDNN libraries for CUDA N2D2 versions (CUDA 11.3 and CuDNN 8)
# Ref: https://medium.com/geekculture/installing-cudnn-and-cuda-toolkit-on-ubuntu-20-04-for-machine-learning-tasks-f41985fcf9b2
if [ -n "$USE_CUDA" ] ; then

    # Install the package for CUDA
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

    CUDA_PKG=cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
    wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/$CUDA_PKG
    sudo dpkg -i $CUDA_PKG
    rm $CUDA_PKG

    # Add keys
    sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub

    # Update the package lists
    sudo apt-get -y update

    # Install the CUDA package
    sudo apt-get -y install cuda

    # Install the runtime package for CuDNN
    CUDNN_PKG=libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/$CUDNN_PKG
    sudo dpkg -i $CUDNN_PKG
    rm $CUDNN_PKG

    # Install the developer package for CuDNN
    CUDNN_DEV_PKG=libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/$CUDNN_DEV_PKG
    sudo dpkg -i $CUDNN_DEV_PKG
    rm $CUDNN_DEV_PKG
    
fi
