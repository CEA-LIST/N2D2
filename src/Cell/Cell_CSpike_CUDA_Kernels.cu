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

#include "Cell/Cell_CSpike_CUDA_Kernels.hpp"


__global__ void cudaIaccumulate_kernel(int* x, int* y, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] += y[i];
    }
}

__global__ void cudaSaccumulate_kernel(float* x, int* y, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] += y[i];
    }
}

__global__ void cudaDaccumulate_kernel(double* x, int* y, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] += y[i];
    }
}


void N2D2::cudaIaccumulate(int* x, int* y, unsigned int size)
{
    cudaIaccumulate_kernel<<<(size + 255) / 256, 256>>>(x, y, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSaccumulate(float* x, int* y, unsigned int size)
{
    cudaSaccumulate_kernel<<<(size + 255) / 256, 256>>>(x, y, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDaccumulate(double* x, int* y, unsigned int size)
{
    cudaDaccumulate_kernel<<<(size + 255) / 256, 256>>>(x, y, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
