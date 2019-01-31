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
#include "CudaUtils.hpp"

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

__global__ void cudaHquantize_kernel(__half* x,
                                     __half* y,
                                     unsigned int size,
                                     __half minVal,
                                     __half maxVal,
                                     unsigned int quantizationLevels,
                                     bool truncate)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (quantizationLevels > 1) {
        const float scaling = (__half2float(maxVal) - __half2float(minVal))
            / (float)(quantizationLevels - 1);

        for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
            const __half clamped = (__hlt(x[i], minVal)) ? minVal :
                                   (__hgt(x[i], maxVal)) ? maxVal :
                                                           x[i];
#else
            const __half clamped
                = (__half2float(x[i]) < __half2float(minVal)) ? minVal :
                  (__half2float(x[i]) > __half2float(maxVal)) ? maxVal :
                                                                x[i];
#endif

            if (truncate) {
                y[i] = __float2half(
                    (int)((__half2float(clamped) - __half2float(minVal))
                               / scaling) * scaling + __half2float(minVal));
            }
            else {
                y[i] = __float2half(
                    (int)round((__half2float(clamped) - __half2float(minVal))
                               / scaling) * scaling + __half2float(minVal));
            }
        }
    }
    else {
        for (unsigned int i = index; i < size; i += stride)
            y[i] = __float2half((__half2float(x[i]) >= 0.0f) ? 1.0f : -1.0f);
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

__global__ void cudaSquantize_kernel(float* x,
                                     float* y,
                                     unsigned int size,
                                     float minVal,
                                     float maxVal,
                                     unsigned int quantizationLevels,
                                     bool truncate)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (quantizationLevels > 1) {
        const float scaling = (maxVal - minVal)
            / (float)(quantizationLevels - 1);

        for (unsigned int i = index; i < size; i += stride) {
            const float clamped = (x[i] < minVal) ? minVal :
                                  (x[i] > maxVal) ? maxVal :
                                                    x[i];

            if (truncate)
                y[i] = (int)((clamped - minVal) / scaling) * scaling + minVal;
            else {
                y[i] = (int)round((clamped - minVal) / scaling)
                        * scaling + minVal;
            }
        }
    }
    else {
        for (unsigned int i = index; i < size; i += stride)
            y[i] = ((x[i] >= 0.0f) ? 1.0f : -1.0f);
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

__global__ void cudaDquantize_kernel(double* x,
                                     double* y,
                                     unsigned int size,
                                     double minVal,
                                     double maxVal,
                                     unsigned int quantizationLevels,
                                     bool truncate)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (quantizationLevels > 1) {
        const double scaling = (maxVal - minVal)
            / (double)(quantizationLevels - 1);

        for (unsigned int i = index; i < size; i += stride) {
            const double clamped = (x[i] < minVal) ? minVal :
                                  (x[i] > maxVal) ? maxVal :
                                                    x[i];

            if (truncate)
                y[i] = (int)((clamped - minVal) / scaling) * scaling + minVal;
            else {
                y[i] = (int)round((clamped - minVal) / scaling)
                        * scaling + minVal;
            }
        }
    }
    else {
        for (unsigned int i = index; i < size; i += stride)
            y[i] = ((x[i] >= 0.0) ? 1.0 : -1.0);
    }
}

__global__ void cudaHscal_kernel(unsigned int size,
                                 __half alpha,
                                 __half *x)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        x[i] = __hmul(alpha, x[i]);
#else
        x[i] = __float2half(__half2float(alpha) * __half2float(x[i]));
#endif
    }
}

__global__ void cudaHaxpy_kernel(unsigned int size,
                                 __half alpha,
                                 const __half *x,
                                 __half *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hadd(__hmul(alpha, x[i]), y[i]);
#else
        y[i] = __float2half(__half2float(alpha) * __half2float(x[i])
                            + __half2float(y[i]));
#endif
    }
}

__global__ void cudaHpow_kernel(unsigned int size,
                                 __half power,
                                 const __half *x,
                                 __half *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = __float2half(powf(__half2float(x[i]), __half2float(power)));
    }
}

__global__ void cudaSpow_kernel(unsigned int size,
                                 float power,
                                 const float *x,
                                 float *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = powf(x[i], power);
    }
}

__global__ void cudaDpow_kernel(unsigned int size,
                                 double power,
                                 const double *x,
                                 double *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = powf(x[i], power);
    }
}

__global__ void cudaHadd_kernel(unsigned int size,
                                 __half value,
                                 const __half *x,
                                 __half *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hadd(x[i], value);
#else
        y[i] = __float2half(__half2float(x[i]) + __half2float(value));
#endif
    }
}

__global__ void cudaSadd_kernel(unsigned int size,
                                 float value,
                                 const float *x,
                                 float *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = x[i] + value;
    }
}

__global__ void cudaDadd_kernel(unsigned int size,
                                 double value,
                                 const double *x,
                                 double *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = x[i] + value;
    }
}

__global__ void cudaHmult_kernel(unsigned int size,
                                 const __half *x1,
                                 const __half *x2,
                                 __half *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul(x1[i], x2[i]);
#else
        y[i] = __float2half(__half2float(x1[i]) + __half2float(x2[i]));
#endif
    }
}

__global__ void cudaSmult_kernel(unsigned int size,
                                 const float *x1,
                                 const float *x2,
                                 float *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = x1[i] * x2[i];
    }
}

__global__ void cudaDmult_kernel(unsigned int size,
                                 const double *x1,
                                 const double *x2,
                                 double *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = x1[i] * x2[i];
    }
}

__global__ void cudaHinv_kernel(unsigned int size,
                                 const __half *x,
                                 __half *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = __float2half(1.0f / __half2float(x[i]));
    }
}

__global__ void cudaSinv_kernel(unsigned int size,
                                 const float *x,
                                 float *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = 1.0f / x[i];
    }
}

__global__ void cudaDinv_kernel(unsigned int size,
                                 const double *x,
                                 double *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = 1.0 / x[i];
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

struct HalfLess : public std::binary_function<__half, __half, bool> {
    __device__ bool operator()(const __half& left, const __half& right) const
    {
#if __CUDA_ARCH__ >= 530
        return __hlt(left, right);
#else
        return (__half2float(left) < __half2float(right));
#endif
    }
};

std::pair<half_float::half, half_float::half>
N2D2::cudaHminMax(half_float::half* x,
                  unsigned int size)
{
    // Compute global min & max value on the full tensor
    thrust::device_ptr<__half> thrustPtr(reinterpret_cast<__half*>(x));
    thrust::pair<thrust::device_vector<__half>::iterator,
                 thrust::device_vector<__half>::iterator> minMaxPair
        = thrust::minmax_element(thrustPtr, thrustPtr + size, HalfLess());

    const __half minVal = *(minMaxPair.first);
    const __half maxVal = *(minMaxPair.second);

    return std::make_pair(reinterpret_cast<const half_float::half&>(minVal),
                          reinterpret_cast<const half_float::half&>(maxVal));
}

void N2D2::cudaHquantize(half_float::half* x,
                         half_float::half* y,
                         unsigned int size,
                         half_float::half minVal,
                         half_float::half maxVal,
                         unsigned int quantizationLevels,
                         bool truncate)
{
    cudaHquantize_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x),
         reinterpret_cast<__half*>(y),
         size,
         reinterpret_cast<__half&>(minVal),
         reinterpret_cast<__half&>(maxVal),
         quantizationLevels,
         truncate);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSclamp(float* x, unsigned int size, float minVal, float maxVal)
{
    cudaSclamp_kernel<<<(size + 255) / 256, 256>>>(x, size, minVal, maxVal);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

std::pair<float, float>
N2D2::cudaSminMax(float* x,
                  unsigned int size)
{
    // Compute global min & max value on the full tensor
    thrust::device_ptr<float> thrustPtr(x);
    thrust::pair<thrust::device_vector<float>::iterator,
                 thrust::device_vector<float>::iterator> minMaxPair
        = thrust::minmax_element(thrustPtr, thrustPtr + size);

    return std::make_pair(*(minMaxPair.first), *(minMaxPair.second));
}

void N2D2::cudaSquantize(float* x,
                         float* y,
                         unsigned int size,
                         float minVal,
                         float maxVal,
                         unsigned int quantizationLevels,
                         bool truncate)
{
    cudaSquantize_kernel<<<(size + 255) / 256, 256>>>
        (x, y, size, minVal, maxVal, quantizationLevels, truncate);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void
N2D2::cudaDclamp(double* x, unsigned int size, double minVal, double maxVal)
{
    cudaDclamp_kernel<<<(size + 255) / 256, 256>>>
        (x, size, minVal, maxVal);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

std::pair<double, double>
N2D2::cudaDminMax(double* x,
                  unsigned int size)
{
    // Compute global min & max value on the full tensor
    thrust::device_ptr<double> thrustPtr(x);
    thrust::pair<thrust::device_vector<double>::iterator,
                 thrust::device_vector<double>::iterator> minMaxPair
        = thrust::minmax_element(thrustPtr, thrustPtr + size);

    return std::make_pair(*(minMaxPair.first), *(minMaxPair.second));
}

void N2D2::cudaDquantize(double* x,
                         double* y,
                         unsigned int size,
                         double minVal,
                         double maxVal,
                         unsigned int quantizationLevels,
                         bool truncate)
{
    cudaDquantize_kernel<<<(size + 255) / 256, 256>>>
        (x, y, size, minVal, maxVal, quantizationLevels, truncate);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHscal(unsigned int size,
                     half_float::half alpha,
                     half_float::half *x)
{
    cudaHscal_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<__half&>(alpha),
        reinterpret_cast<__half*>(x));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHaxpy(unsigned int size,
                     half_float::half alpha,
                     const half_float::half *x,
                     half_float::half *y)
{
    cudaHaxpy_kernel<<<(size + 255) / 256, 256>>>
        (size,
         reinterpret_cast<__half&>(alpha),
         reinterpret_cast<const __half*>(x),
         reinterpret_cast<__half*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHpow(unsigned int size,
                      half_float::half power,
                      const half_float::half *x,
                      half_float::half *y)
{
    cudaHpow_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<__half&>(power),
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<__half*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSpow(unsigned int size,
                      float power,
                      const float *x,
                      float *y)
{
    cudaSpow_kernel<<<(size + 255) / 256, 256>>>(size, power, x, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDpow(unsigned int size,
                      double power,
                      const double *x,
                      double *y)
{
    cudaDpow_kernel<<<(size + 255) / 256, 256>>>(size, power, x, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHadd(unsigned int size,
                    half_float::half value,
                    const half_float::half *x,
                    half_float::half *y)
{
    cudaHadd_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<__half&>(value),
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<__half*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSadd(unsigned int size,
                      float value,
                      const float *x,
                      float *y)
{
    cudaSadd_kernel<<<(size + 255) / 256, 256>>>(size, value, x, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDadd(unsigned int size,
                      double value,
                      const double *x,
                      double *y)
{
    cudaDadd_kernel<<<(size + 255) / 256, 256>>>(size, value, x, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


void N2D2::cudaHmult(unsigned int size,
                      const half_float::half *x1,
                      const half_float::half *x2,
                      half_float::half *y)
{
    cudaHmult_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<const __half*>(x1),
        reinterpret_cast<const __half*>(x2),
        reinterpret_cast<__half*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSmult(unsigned int size,
                      const float *x1,
                      const float *x2,
                      float *y)
{
    cudaSmult_kernel<<<(size + 255) / 256, 256>>>(size, x1, x2, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDmult(unsigned int size,
                      const double *x1,
                      const double *x2,
                      double *y)
{
    cudaDmult_kernel<<<(size + 255) / 256, 256>>>(size, x1, x2, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHinv(unsigned int size,
                    const half_float::half *x,
                    half_float::half *y)
{
    cudaHinv_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<__half*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSinv(unsigned int size,
                      const float *x,
                      float *y)
{
    cudaSinv_kernel<<<(size + 255) / 256, 256>>>(size, x, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDinv(unsigned int size,
                      const double *x,
                      double *y)
{
    cudaDinv_kernel<<<(size + 255) / 256, 256>>>(size, x, y);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
