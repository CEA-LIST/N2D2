/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
*/

#include "Solver/SGDSolver_CUDA_Kernels.hpp"

__global__ void
cudaSclamp_kernel(float* x, unsigned int size, float minVal, float maxVal)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] = (x[i] < minVal) ? minVal :
               (x[i] > maxVal) ? maxVal :
                                 x[i];
    }
}

__global__ void cudaSquantize_kernel(float* y,
                                     float* x,
                                     unsigned int size,
                                     unsigned int quantizationLevels)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = (quantizationLevels > 1)
                       ? (int)round((quantizationLevels - 1) * x[i])
                         / (float)(quantizationLevels - 1)
                       : ((x[i] >= 0) ? 1 : -1);
    }
}

__global__ void
cudaDclamp_kernel(double* x, unsigned int size, double minVal, double maxVal)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] = (x[i] < minVal) ? minVal :
               (x[i] > maxVal) ? maxVal :
                                 x[i];
    }
}

__global__ void cudaDquantize_kernel(double* y,
                                     double* x,
                                     unsigned int size,
                                     unsigned int quantizationLevels)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = (quantizationLevels > 1)
                       ? (int)round((quantizationLevels - 1) * x[i])
                         / (double)(quantizationLevels - 1)
                       : ((x[i] >= 0) ? 1 : -1);
    }
}

void N2D2::cudaSclamp(float* x, unsigned int size, float minVal, float maxVal)
{
    cudaSclamp_kernel<<<(size + 255) / 256, 256>>>(x, size, minVal, maxVal);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSquantize(float* y,
                         float* x,
                         unsigned int size,
                         unsigned int quantizationLevels)
{
    cudaSquantize_kernel<<<(size + 255) / 256, 256>>>
        (y, x, size, quantizationLevels);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void
N2D2::cudaDclamp(double* x, unsigned int size, double minVal, double maxVal)
{
    cudaDclamp_kernel<<<(size + 255) / 256, 256>>>
        (x, size, minVal, maxVal);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDquantize(double* y,
                         double* x,
                         unsigned int size,
                         unsigned int quantizationLevels)
{
    cudaDquantize_kernel<<<(size + 255) / 256, 256>>>
        (y, x, size, quantizationLevels);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
