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
cudaHclamp_kernel(__half* x, unsigned int size, __half minVal, __half maxVal)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        x[i] = (__hlt(x[i], minVal)) ? minVal :
               (__hgt(x[i], maxVal)) ? maxVal :
                                       x[i];
#else
        x[i] = (__half2float(x[i]) < __half2float(minVal)) ? minVal :
               (__half2float(x[i]) > __half2float(maxVal)) ? maxVal :
                                                             x[i];
#endif
    }
}

__global__ void cudaHquantize_kernel(__half* y,
                                     __half* x,
                                     unsigned int size,
                                     unsigned int quantizationLevels)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = (quantizationLevels > 1)
                       ? __float2half((int)round((quantizationLevels - 1) * __half2float(x[i]))
                         / (float)(quantizationLevels - 1))
                       : __int2half_rn((__half2float(x[i]) >= 0.0f) ? 1 : -1);
    }
}

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

__global__ void cudaHscal_kernel(int n,
                                 const __half *alpha,
                                 __half *x)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < n; i += stride) {
#if __CUDA_ARCH__ >= 530
        x[i] = __hmul((*alpha), x[i]);
#else
        x[i] = __float2half(__half2float(*alpha) * __half2float(x[i]));
#endif
    }
}

__global__ void cudaHaxpy_kernel(int n,
                                 const __half *alpha,
                                 const __half *x,
                                 __half *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < n; i += stride) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hadd(__hmul((*alpha), x[i]), y[i]);
#else
        y[i] = __float2half(__half2float(*alpha) * __half2float(x[i])
                            + __half2float(y[i]));
#endif
    }
}

void N2D2::cudaHclamp(half_float::half* x, unsigned int size,
                      half_float::half minVal, half_float::half maxVal)
{
    cudaHclamp_kernel<<<(size + 255) / 256, 256>>>(reinterpret_cast<__half*>(x),
                                            size,
                                            reinterpret_cast<__half&>(minVal),
                                            reinterpret_cast<__half&>(maxVal));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHquantize(half_float::half* y,
                         half_float::half* x,
                         unsigned int size,
                         unsigned int quantizationLevels)
{
    cudaHquantize_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(y),
         reinterpret_cast<__half*>(x),
         size, quantizationLevels);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
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

void N2D2::cudaHscal(int n,
                     const half_float::half *alpha,
                     half_float::half *x)
{
    cudaHscal_kernel<<<(n + 255) / 256, 256>>>
        (n,
        reinterpret_cast<const __half*>(alpha),
        reinterpret_cast<__half*>(x));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHaxpy(int n,
                     const half_float::half *alpha,
                     const half_float::half *x,
                     half_float::half *y)
{
    cudaHaxpy_kernel<<<(n + 255) / 256, 256>>>
        (n,
         reinterpret_cast<const __half*>(alpha),
         reinterpret_cast<const __half*>(x),
         reinterpret_cast<__half*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
