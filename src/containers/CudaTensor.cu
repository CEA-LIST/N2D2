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

#include "containers/CudaTensor.hpp"

template <typename T>
void N2D2::thrust_fill(T* devData, size_t size, T value)
{
    thrust::device_ptr<T> thrustPtr(devData);
    thrust::fill(thrustPtr, thrustPtr + size, value);
}

// Floating point numbers
template void N2D2::thrust_fill<double>(double* devData,
                                        size_t size, double value);
template void N2D2::thrust_fill<float>(float* devData,
                                       size_t size, float value);
template void N2D2::thrust_fill<half_float::half>(
    half_float::half* devData, size_t size, half_float::half value);

// Signed integers
template void N2D2::thrust_fill<char>(char* devData, size_t size, char value);
template void N2D2::thrust_fill<short>(short* devData, size_t size,
                                       short value);
template void N2D2::thrust_fill<int>(int* devData, size_t size, int value);
template void N2D2::thrust_fill<long long int>(
    long long int* devData, size_t size, long long int value);

// Unsigned integers
template void N2D2::thrust_fill<unsigned char>(unsigned char* devData,
                                               size_t size,
                                               unsigned char value);
template void N2D2::thrust_fill<unsigned short>(unsigned short* devData,
                                                size_t size,
                                                unsigned short value);
template void N2D2::thrust_fill<unsigned int>(unsigned int* devData,
                                              size_t size, unsigned int value);
template void N2D2::thrust_fill<unsigned long long int>(
    unsigned long long int* devData, size_t size, unsigned long long int value);

template <>
void N2D2::thrust_copy(double* srcData, float* dstData, size_t size)
{
    thrust::device_ptr<double> thrustSrcPtr(srcData);
    thrust::device_ptr<float> thrustDstPtr(dstData);
    thrust::copy(thrustSrcPtr, thrustSrcPtr + size, thrustDstPtr);
}

__global__ void
cudaCopyDToH_kernel(double* srcData,
                    __half* dstData,
                    size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dstData[i] = __float2half((float)srcData[i]);
    }
}

template <>
void N2D2::thrust_copy(double* srcData, half_float::half* dstData, size_t size)
{
    cudaCopyDToH_kernel<<<(size + 255) / 256, 256>>>
        (srcData, reinterpret_cast<__half*>(dstData), size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <>
void N2D2::thrust_copy(float* srcData, double* dstData, size_t size)
{
    thrust::device_ptr<float> thrustSrcPtr(srcData);
    thrust::device_ptr<double> thrustDstPtr(dstData);
    thrust::copy(thrustSrcPtr, thrustSrcPtr + size, thrustDstPtr);
}

__global__ void
cudaCopyFToH_kernel(float* srcData,
                    __half* dstData,
                    size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dstData[i] = __float2half(srcData[i]);
    }
}

template <>
void N2D2::thrust_copy(float* srcData, half_float::half* dstData, size_t size)
{
    cudaCopyFToH_kernel<<<(size + 255) / 256, 256>>>
        (srcData, reinterpret_cast<__half*>(dstData), size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

__global__ void
cudaCopyHToD_kernel(__half* srcData,
                    double* dstData,
                    size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dstData[i] = (double)__half2float(srcData[i]);
    }
}

template <>
void N2D2::thrust_copy(half_float::half* srcData, double* dstData, size_t size)
{
    cudaCopyHToD_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(srcData), dstData, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

__global__ void
cudaCopyHToF_kernel(__half* srcData,
                    float* dstData,
                    size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dstData[i] = __half2float(srcData[i]);
    }
}

template <>
void N2D2::thrust_copy(half_float::half* srcData, float* dstData, size_t size)
{
    cudaCopyHToF_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(srcData), dstData, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
