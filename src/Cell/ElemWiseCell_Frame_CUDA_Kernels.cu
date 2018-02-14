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

__global__ void cudaSSqrt_kernel(unsigned int size,
                                 float* data)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        data[i] = sqrt(data[i]);
}

__global__ void cudaSMult_kernel(unsigned int size,
                                 float* a,
                                 float* b,
                                 const float beta,
                                 float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        result[i] = a[i] * b[i] + beta * result[i];
}

__global__ void cudaSScale_kernel(unsigned int size,
                                  float* input,
                                  const float scale,
                                  const float beta,
                                  float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        result[i] = input[i] * scale + beta * result[i];
}

__global__ void cudaSScaleAbs_kernel(unsigned int size,
                                     float* input,
                                     const float scale,
                                     const float beta,
                                     float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        result[i] = fabs(input[i]) * scale + beta * result[i];
}

__global__ void cudaSScaleSign_kernel(unsigned int size,
                                      float* input,
                                      float* sign,
                                      const float scale,
                                      const float beta,
                                      float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        const float sgn = (sign[i] >= 0) ? 1.0f : -1.0f;
        result[i] = input[i] * sgn * scale + beta * result[i];
    }
}

__global__ void cudaSScaleSquare_kernel(unsigned int size,
                                        float* input,
                                        const float scale,
                                        const float beta,
                                        float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        result[i] = input[i] * input[i] * scale + beta * result[i];
}

__global__ void cudaSMaxForward_kernel(unsigned int size,
                                       float* input,
                                       float* maxVal,
                                       const unsigned int idx,
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
                                        const unsigned int idx,
                                        unsigned int* argMax,
                                        const float beta,
                                        float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        result[i] = (argMax[i] == idx) ? (diffInput[i] + beta * result[i])
                                       : beta * result[i];
    }
}

__global__ void cudaSEuclideanSumBackward_kernel(unsigned int size,
                                                 float* diffInput,
                                                 float* input,
                                                 float* output,
                                                 const float scale,
                                                 const float beta,
                                                 float* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        result[i] = (output[i] != 0.0f)
            ? diffInput[i] * scale * (input[i] / output[i]) + beta * result[i]
            : beta * result[i];
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

void N2D2::cudaSSqrt(unsigned int size,
                     float* data)
{
    cudaSSqrt_kernel<<<(size + 255) / 256, 256>>>(size, data);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSMult(unsigned int size,
                     float* a,
                     float* b,
                     const float beta,
                     float* result)
{
    cudaSMult_kernel<<<(size + 255) / 256, 256>>>(size, a, b, beta, result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSScale(unsigned int size,
                      float* input,
                      const float scale,
                      const float beta,
                      float* result)
{
    cudaSScale_kernel<<<(size + 255) / 256, 256>>>(size,
                                                   input,
                                                   scale,
                                                   beta,
                                                   result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSScaleAbs(unsigned int size,
                         float* input,
                         const float scale,
                         const float beta,
                         float* result)
{
    cudaSScaleAbs_kernel<<<(size + 255) / 256, 256>>>(size,
                                                      input,
                                                      scale,
                                                      beta,
                                                      result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSScaleSign(unsigned int size,
                          float* input,
                          float* sign,
                          const float scale,
                          const float beta,
                          float* result)
{
    cudaSScaleSign_kernel<<<(size + 255) / 256, 256>>>(size,
                                                       input,
                                                       sign,
                                                       scale,
                                                       beta,
                                                       result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSScaleSquare(unsigned int size,
                            float* input,
                            const float scale,
                            const float beta,
                            float* result)
{
    cudaSScaleSquare_kernel<<<(size + 255) / 256, 256>>>(size,
                                                         input,
                                                         scale,
                                                         beta,
                                                         result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSMaxForward(unsigned int size,
                           float* input,
                           float* maxVal,
                           const unsigned int idx,
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
                            const unsigned int idx,
                            unsigned int* argMax,
                            const float beta,
                            float* result)
{
    cudaSMaxBackward_kernel<<<(size + 255) / 256, 256>>>(size,
                                                         diffInput,
                                                         idx,
                                                         argMax,
                                                         beta,
                                                         result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSEuclideanSumBackward(unsigned int size,
                                     float* diffInput,
                                     float* input,
                                     float* output,
                                     const float scale,
                                     const float beta,
                                     float* result)
{
    cudaSEuclideanSumBackward_kernel<<<(size + 255) / 256, 256>>>(size,
                                                                  diffInput,
                                                                  input,
                                                                  output,
                                                                  scale,
                                                                  beta,
                                                                  result);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
