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

#include "Cell/ElemWiseCell_Frame_CUDA_Kernels.hpp"

__global__ void cudaUZeroInit_kernel(unsigned int size,
                                     unsigned int* data)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        data[i] = 0U;
}

__global__ void cudaSZeroInit_kernel(unsigned int size,
                                     float* data)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        data[i] = 0.0f;
}

__global__ void cudaSMult_kernel(unsigned int size,
                                 float* a,
                                 float* b,
                                 float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        result[i] = a[i] * b[i];
}

__global__ void cudaSScale_kernel(unsigned int size,
                                  float* input,
                                  float* scale,
                                  float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        result[i] = input[i] * (*scale);
}

__global__ void cudaSMaxForward_kernel(unsigned int size,
                                       float* input,
                                       float* maxVal,
                                       unsigned int idx,
                                       unsigned int* argMax)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (input[i] > maxVal[i]) {
            maxVal[i] = input[i];
            argMax[i] = idx;
        }
    }
}

__global__ void cudaSMaxBackward_kernel(unsigned int size,
                                        float* diffInput,
                                        unsigned int idx,
                                        unsigned int* argMax,
                                        float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        result[i] = (argMax[i] == idx) ? diffInput[i] : 0.0f;
    }
}

void N2D2::cudaUZeroInit(unsigned int size,
                         unsigned int* data)
{
    cudaUZeroInit_kernel<<<(size + 255) / 256, 256>>>(size, data);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSZeroInit(unsigned int size,
                         float* data)
{
    cudaSZeroInit_kernel<<<(size + 255) / 256, 256>>>(size, data);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSMult(unsigned int size,
                     float* a,
                     float* b,
                     float* result)
{
    cudaSMult_kernel<<<(size + 255) / 256, 256>>>(size, a, b, result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSScale(unsigned int size,
                      float* input,
                      float* scale,
                      float* result)
{
    cudaSScale_kernel<<<(size + 255) / 256, 256>>>(size, input, scale, result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSMaxForward(unsigned int size,
                           float* input,
                           float* maxVal,
                           unsigned int idx,
                           unsigned int* argMax)
{
    cudaSMaxForward_kernel<<<(size + 255) / 256, 256>>>(size,
                                                        input,
                                                        maxVal,
                                                        idx,
                                                        argMax);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSMaxBackward(unsigned int size,
                            float* diffInput,
                            unsigned int idx,
                            unsigned int* argMax,
                            float* result)
{
    cudaSMaxBackward_kernel<<<(size + 255) / 256, 256>>>(size,
                                                         diffInput,
                                                         idx,
                                                         argMax,
                                                         result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
