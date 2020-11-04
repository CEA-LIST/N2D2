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

#include "Activation/Activation_CUDA_Kernels.hpp"

// LeakyRectifier
__global__ void cudaHRectifier_propagate_kernel(__half* x,
                                                __half* y,
                                                unsigned int size,
                                                __half leakSlope,
                                                __half clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        __half value = x[i];

        if (__half2float(clipping) > 0.0f) {
#if __CUDA_ARCH__ >= 530
            y[i] = (__half2float(value) > 0.0f)
                ? ((__hlt(value, clipping))
                    ? value
                    : clipping)
                : __hmul(leakSlope, value);
#else
            y[i] = (__half2float(value) > 0.0f)
                ? ((__half2float(value) < __half2float(clipping))
                    ? value
                    : clipping)
                : __float2half(__half2float(leakSlope) * __half2float(value));
#endif
        }
        else
#if __CUDA_ARCH__ >= 530
            y[i] = (__half2float(value) > 0.0f) ? value : __hmul(leakSlope, value);
#else
            y[i] = (__half2float(value) > 0.0f) ? value
                : __float2half(__half2float(leakSlope) * __half2float(value));
#endif
    }
}

__global__ void cudaSRectifier_propagate_kernel(float* x,
                                                float* y,
                                                unsigned int size,
                                                float leakSlope,
                                                float clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        float value = x[i];

        if (clipping > 0.0f)
            y[i] = (value > 0.0f) ? min(value, clipping) : leakSlope * value;
        else
            y[i] = (value > 0.0f) ? value : leakSlope * value;
    }
}

__global__ void cudaDRectifier_propagate_kernel(double* x,
                                                double* y,
                                                unsigned int size,
                                                double leakSlope,
                                                double clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        double value = x[i];

        if (clipping > 0.0)
            y[i] = (value > 0.0) ? min(value, clipping) : leakSlope * value;
        else
            y[i] = (value > 0.0) ? value : leakSlope * value;
    }
}

__global__ void cudaHRectifier_backPropagate_kernel(__half* x,
                                                    __half* dx,
                                                    unsigned int size,
                                                    __half leakSlope,
                                                    __half clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (__half2float(clipping) > 0.0f) {
#if __CUDA_ARCH__ >= 530
            dx[i] = (__hgt(x[i], clipping))
                ? __float2half(0.0f)
                : (__half2float(x[i]) > 0.0f)
                    ? dx[i]
                    : __hmul(leakSlope, dx[i]);
#else
            dx[i] = (__half2float(x[i]) > __half2float(clipping))
                ? __float2half(0.0f)
                : (__half2float(x[i]) > 0.0f)
                    ? dx[i]
                    : __float2half(__half2float(leakSlope)
                                   * __half2float(dx[i]));
#endif
        }
        else {
#if __CUDA_ARCH__ >= 530
            dx[i] = (__half2float(x[i]) > 0.0f) ? dx[i]
                                                : __hmul(leakSlope, dx[i]);
#else
            dx[i] = (__half2float(x[i]) > 0.0f) ? dx[i]
                : __float2half(__half2float(leakSlope) * __half2float(dx[i]));
#endif
        }
    }
}

__global__ void cudaSRectifier_backPropagate_kernel(float* x,
                                                    float* dx,
                                                    unsigned int size,
                                                    float leakSlope,
                                                    float clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (clipping > 0.0) {
            dx[i] *= (x[i] > clipping) ? 0.0f : (x[i] > 0.0f)
                                       ? 1.0f
                                       : leakSlope;
        }
        else
            dx[i] *= (x[i] > 0.0f) ? 1.0f : leakSlope;
    }
}

__global__ void cudaDRectifier_backPropagate_kernel(double* x,
                                                    double* dx,
                                                    unsigned int size,
                                                    double leakSlope,
                                                    double clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (clipping > 0.0) {
            dx[i] *= (x[i] > clipping) ? 0.0 : (x[i] > 0.0)
                                       ? 1.0
                                       : leakSlope;
        }
        else
            dx[i] *= (x[i] > 0.0) ? 1.0 : leakSlope;
    }
}

// Saturation
__global__ void cudaHSaturation_propagate_kernel(__half* x,
                                                 __half* y,
                                                 unsigned int size,
                                                 __half threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        __half value = x[i];

        if (__half2float(threshold) != 0.0f) {
#if __CUDA_ARCH__ >= 530
            y[i] = (__hlt(value, __hneg(threshold))) ? __hneg(threshold)
                 : (__hgt(value, threshold)) ? threshold
                 : value;
#else
            y[i] = (__half2float(value) < -__half2float(threshold))
                 ? __float2half(-__half2float(threshold))
                    : (__half2float(value) > __half2float(threshold)) ? threshold
                    : value;
#endif
        }
    }
}

__global__ void cudaSSaturation_propagate_kernel(float* x,
                                                 float* y,
                                                 unsigned int size,
                                                 float threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        float value = x[i];

        if (threshold != 0.0f) {
            y[i] = (value < -threshold) ? -threshold
                 : (value > threshold) ? threshold
                 : value;
        }
    }
}

__global__ void cudaDSaturation_propagate_kernel(double* x,
                                                 double* y,
                                                 unsigned int size,
                                                 double threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        double value = x[i];

        if (threshold != 0.0) {
            y[i] = (value < -threshold) ? -threshold
                 : (value > threshold) ? threshold
                 : value;
        }
    }
}

__global__ void
cudaHSaturation_backPropagate_kernel(__half* x,
                                     __half* dx,
                                     unsigned int size,
                                     __half threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (__half2float(threshold) != 0.0f) {
#if __CUDA_ARCH__ >= 530
            dx[i] = (__hgt(x[i], __hneg(threshold)) && __hlt(x[i], threshold))
                ? dx[i] : __float2half(0.0f);
#else
            dx[i] = (__half2float(x[i]) > -__half2float(threshold)
                     && __half2float(x[i]) < __half2float(threshold))
                ? dx[i] : __float2half(0.0f);
#endif
        }
    }
}

__global__ void
cudaSSaturation_backPropagate_kernel(float* x,
                                     float* dx,
                                     unsigned int size,
                                     float threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (threshold != 0.0f) {
            dx[i] *= (x[i] > -threshold && x[i] < threshold)
                ? 1.0f : 0.0f;
        }
    }
}

__global__ void
cudaDSaturation_backPropagate_kernel(double* x,
                                     double* dx,
                                     unsigned int size,
                                     double threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (threshold != 0.0) {
            dx[i] *= (x[i] > -threshold && x[i] < threshold)
                ? 1.0 : 0.0;
        }
    }
}

// Softplus
__global__ void cudaHSoftplus_propagate_kernel(__half* x,
                                               __half* y,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
// hexp and hlog are only available since CUDA 8.0
#if __CUDA_ARCH__ >= 530 && defined(CUDART_VERSION) && CUDART_VERSION >= 8000
        y[i] = hlog(__hadd(__float2half(1.0f), hexp(x[i])));
#else
        y[i] = __float2half(log(1.0f + exp(__half2float(x[i]))));
#endif
    }
}

__global__ void cudaSSoftplus_propagate_kernel(float* x,
                                               float* y,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = log(1.0f + exp(x[i]));
    }
}

__global__ void cudaDSoftplus_propagate_kernel(double* x,
                                               double* y,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = log(1.0 + exp(x[i]));
    }
}

__global__ void
cudaHSoftplus_backPropagate_kernel(__half* x, __half* dx, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
// hexp and hlog are only available since CUDA 8.0
#if __CUDA_ARCH__ >= 530 && defined(CUDART_VERSION) && CUDART_VERSION >= 8000
        dx[i] = __hmul(dx[i], (__hsub(__float2half(1.0f), hexp(__hneg(x[i])))));
#else
        dx[i] = __float2half(__half2float(dx[i])
                             * (1.0f - exp(-__half2float(x[i]))));
#endif
    }
}

__global__ void
cudaSSoftplus_backPropagate_kernel(float* x, float* dx, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dx[i] *= (1.0f - exp(-x[i]));
    }
}

__global__ void
cudaDSoftplus_backPropagate_kernel(double* x, double* dx, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dx[i] *= (1.0 - exp(-x[i]));
    }
}

// Swish
__global__ void cudaHSwish_propagate_kernel(__half* x,
                                            __half* y,
                                            __half* sigmoid,
                                            unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        const __half sig = __hdiv(__float2half(1.0f),
                                  __hadd(__float2half(1.0f),
                                         hexp(__hneg(x[i]))));
#else
        const __half sig
            = __float2half(1.0f / (1.0f + exp(-__half2float(x[i]))));
#endif

        sigmoid[i] = sig;
#if __CUDA_ARCH__ >= 530
        y[i] = __hmul(x[i], sig);
#else
        y[i] = __float2half(__half2float(x[i]) * __half2float(sig));
#endif
    }
}

__global__ void cudaSSwish_propagate_kernel(float* x,
                                            float* y,
                                            float* sigmoid,
                                            unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        const float sig = 1.0f / (1.0f + exp(-x[i]));
        sigmoid[i] = sig;
        y[i] = x[i] * sig;
    }
}

__global__ void cudaDSwish_propagate_kernel(double* x,
                                            double* y,
                                            double* sigmoid,
                                            unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        const double sig = 1.0 / (1.0 + exp(-x[i]));
        sigmoid[i] = sig;
        y[i] = x[i] * sig;
    }
}

__global__ void cudaHSwish_backPropagate_kernel(__half* x,
                                                __half* dx,
                                                __half* sigmoid,
                                                unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        dx[i] = __hmul(dx[i, __hadd(sigmoid[i],
                             __hmul(x[i],
                                    __hsub(__float2half(1.0f), sigmoid[i]))));
#else
        const float sig = __half2float(sigmoid[i]);
        dx[i] = __float2half(__half2float(dx[i]) 
            * (sig + __half2float(x[i]) * (1.0f - sig)));
#endif
    }
}

__global__ void cudaSSwish_backPropagate_kernel(float* x,
                                                float* dx,
                                                float* sigmoid,
                                                unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dx[i] *= sigmoid[i] + x[i] * (1.0f - sigmoid[i]);
    }
}

__global__ void cudaDSwish_backPropagate_kernel(double* x,
                                                double* dx,
                                                double* sigmoid,
                                                unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dx[i] *= sigmoid[i] + x[i] * (1.0 - sigmoid[i]);
    }
}

// Rectifier
void N2D2::cudaHRectifier_propagate(half_float::half* x,
                                    half_float::half* y,
                                    unsigned int size,
                                    half_float::half leakSlope,
                                    half_float::half clipping)
{
    cudaHRectifier_propagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x),
         reinterpret_cast<__half*>(y),
         size,
         reinterpret_cast<__half&>(leakSlope),
         reinterpret_cast<__half&>(clipping));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSRectifier_propagate(float* x,
                                    float* y,
                                    unsigned int size,
                                    float leakSlope,
                                    float clipping)
{
    cudaSRectifier_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, y, size, leakSlope, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDRectifier_propagate(double* x,
                                    double* y,
                                    unsigned int size,
                                    double leakSlope,
                                    double clipping)
{
    cudaDRectifier_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, y, size, leakSlope, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHRectifier_backPropagate(half_float::half* x,
                                        half_float::half* dx,
                                        unsigned int size,
                                        half_float::half leakSlope,
                                        half_float::half clipping)
{
    cudaHRectifier_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x),
         reinterpret_cast<__half*>(dx),
         size,
         reinterpret_cast<__half&>(leakSlope),
         reinterpret_cast<__half&>(clipping));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSRectifier_backPropagate(float* x,
                                        float* dx,
                                        unsigned int size,
                                        float leakSlope,
                                        float clipping)
{
    cudaSRectifier_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, leakSlope, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDRectifier_backPropagate(double* x,
                                        double* dx,
                                        unsigned int size,
                                        double leakSlope,
                                        double clipping)
{
    cudaDRectifier_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, leakSlope, clipping);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Saturation
void N2D2::cudaHSaturation_propagate(half_float::half* x,
                                     half_float::half* y,
                                     unsigned int size,
                                     half_float::half threshold)
{
    cudaHSaturation_propagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x),
         reinterpret_cast<__half*>(y),
         size,
         reinterpret_cast<__half&>(threshold));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSaturation_propagate(float* x,
                                     float* y,
                                     unsigned int size,
                                     float threshold)
{
    cudaSSaturation_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, y, size, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSaturation_propagate(double* x,
                                     double* y,
                                     unsigned int size,
                                     double threshold)
{
    cudaDSaturation_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, y, size, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHSaturation_backPropagate(half_float::half* x,
                                         half_float::half* dx,
                                         unsigned int size,
                                         half_float::half threshold)
{
    cudaHSaturation_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x),
         reinterpret_cast<__half*>(dx),
         size,
         reinterpret_cast<__half&>(threshold));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSaturation_backPropagate(float* x,
                                         float* dx,
                                         unsigned int size,
                                         float threshold)
{
    cudaSSaturation_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void
N2D2::cudaDSaturation_backPropagate(double* x,
                                    double* dx,
                                    unsigned int size,
                                    double threshold)
{
    cudaDSaturation_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size, threshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Softplus
void N2D2::cudaHSoftplus_propagate(half_float::half* x,
                                   half_float::half* y,
                                   unsigned int size)
{
    cudaHSoftplus_propagate_kernel<<<(size + 255) / 256, 256>>>(
        reinterpret_cast<__half*>(x),
        reinterpret_cast<__half*>(y),
        size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSoftplus_propagate(float* x, float* y, unsigned int size)
{
    cudaSSoftplus_propagate_kernel<<<(size + 255) / 256, 256>>>(x, y, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSoftplus_propagate(double* x, double* y, unsigned int size)
{
    cudaDSoftplus_propagate_kernel<<<(size + 255) / 256, 256>>>(x, y, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHSoftplus_backPropagate(half_float::half* x,
                                       half_float::half* dx,
                                       unsigned int size)
{
    cudaHSoftplus_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x), reinterpret_cast<__half*>(dx), size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSoftplus_backPropagate(float* x, float* dx, unsigned int size)
{
    cudaSSoftplus_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSoftplus_backPropagate(double* x, double* dx, unsigned int size)
{
    cudaDSoftplus_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Swish
void N2D2::cudaHSwish_propagate(half_float::half* x,
                                half_float::half* y,
                                half_float::half* sigmoid,
                                unsigned int size)
{
    cudaHSwish_propagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x),
         reinterpret_cast<__half*>(y),
         reinterpret_cast<__half*>(sigmoid),
         size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSwish_propagate(float* x,
                                float* y,
                                float* sigmoid,
                                unsigned int size)
{
    cudaSSwish_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, y, sigmoid, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSwish_propagate(double* x,
                                double* y,
                                double* sigmoid,
                                unsigned int size)
{
    cudaDSwish_propagate_kernel<<<(size + 255) / 256, 256>>>
        (x, y, sigmoid, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaHSwish_backPropagate(half_float::half* x,
                                    half_float::half* dx,
                                    half_float::half* sigmoid,
                                    unsigned int size)
{
    cudaHSwish_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<__half*>(x),
         reinterpret_cast<__half*>(dx),
         reinterpret_cast<__half*>(sigmoid),
         size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSSwish_backPropagate(float* x,
                                    float* dx,
                                    float* sigmoid,
                                    unsigned int size)
{
    cudaSSwish_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, sigmoid, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDSwish_backPropagate(double* x,
                                    double* dx,
                                    double* sigmoid,
                                    unsigned int size)
{
    cudaDSwish_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (x, dx, sigmoid, size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
