/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Vincent TEMPLIER (vincent.templier@cea.fr)

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

#include "Cell/BatchNormCell_Frame_CUDA_Kernels.hpp"
#include "CudaUtils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>
#include <thrust/iterator/constant_iterator.h>

__global__ void
cudaDivH_kernel(__half* srcData,
                size_t size,
                int value)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    float div_value = 1/(float)value;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530 && defined(CUDART_VERSION) && CUDART_VERSION >= 8000
        srcData[i] = __hmul(srcData[i], __float2half(div_value));
#else
        srcData[i] = __float2half(__half2float(srcData[i]) * div_value);
#endif
    }
}

template <>
void N2D2::thrust_div(half_float::half* srcData,
                      size_t size, 
                      int value)
{
    cudaDivH_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(srcData), size, value);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <>
void N2D2::thrust_div(float* srcData, size_t size, int value)
{
    thrust::device_ptr<float> thrustSrcPtr(srcData);
    thrust::transform(
        thrustSrcPtr, 
        thrustSrcPtr + size,
        thrust::make_constant_iterator((float)value), 
        thrustSrcPtr,
        thrust::divides<float>());
}

template <>
void N2D2::thrust_div(double* srcData, size_t size, int value)
{
    thrust::device_ptr<double> thrustSrcPtr(srcData);
    thrust::transform(
        thrustSrcPtr, 
        thrustSrcPtr + size,
        thrust::make_constant_iterator((double)value), 
        thrustSrcPtr,
        thrust::divides<double>());
}

__global__ void
cudaCombVarH_kernel(__half* var,
                    __half* mean,
                    __half* copyMean,
                    size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        var[i] = __float2half(__half2float(var[i]) 
                + std::pow(__half2float(copyMean[i]) - __half2float(mean[i]),2));
    }
}

__global__ void
cudaCombVarF_kernel(float* var,
                    float* mean,
                    float* copyMean,
                    size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        var[i] = var[i] + std::pow(copyMean[i] - mean[i],2);
    }
}

__global__ void
cudaCombVarD_kernel(double* var,
                    double* mean,
                    double* copyMean,
                    size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        var[i] = var[i] + std::pow(copyMean[i] - mean[i],2);
    }
}

template <>
void N2D2::thrust_combinedVar(half_float::half* var,
                              half_float::half* mean,
                              half_float::half* copyMean,
                              size_t size)
{
    cudaCombVarH_kernel<<<(size + 255) / 256, 256>>> (
        reinterpret_cast<__half*>(var),
        reinterpret_cast<__half*>(mean),
        reinterpret_cast<__half*>(copyMean),
        size
    );
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <>
void N2D2::thrust_combinedVar(float* var,
                              float* mean,
                              float* copyMean,  
                              size_t size)
{
    cudaCombVarF_kernel<<<(size + 255) / 256, 256>>> 
        (var, mean, copyMean, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <>
void N2D2::thrust_combinedVar(double* var,
                              double* mean,
                              double* copyMean,  
                              size_t size)
{
    cudaCombVarD_kernel<<<(size + 255) / 256, 256>>> 
        (var, mean, copyMean, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}