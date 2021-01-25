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
template <class T>
__global__ void cudaRectifier_propagate_kernel(T* x,
                                                T* y,
                                                unsigned int size,
                                                T leakSlope,
                                                T clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        T value = x[i];

        if (clipping > 0.0f)
            y[i] = (value > 0.0f) ? min(value, clipping) : leakSlope * value;
        else
            y[i] = (value > 0.0f) ? value : leakSlope * value;
    }
}

template <>
__global__ void cudaRectifier_propagate_kernel<__half>(__half* x,
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

template <class T>
__global__ void cudaRectifier_backPropagate_kernel(T* y,
                                                    T* dx,
                                                    T* dy,
                                                    unsigned int size,
                                                    T leakSlope,
                                                    T clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (clipping > 0.0) {
            dy[i] = dx[i] * ((y[i] == clipping) ? 0.0f : (y[i] > 0.0f)
                                       ? 1.0f
                                       : leakSlope);
        }
        else
            dy[i] = dx[i] * ((y[i] > 0.0f) ? 1.0f : leakSlope);
    }
}

template <>
__global__ void cudaRectifier_backPropagate_kernel<__half>(__half* y,
                                                    __half* dx,
                                                    __half* dy,
                                                    unsigned int size,
                                                    __half leakSlope,
                                                    __half clipping)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (__half2float(clipping) > 0.0f) {
#if __CUDA_ARCH__ >= 530
            dy[i] = (__heq(y[i], clipping))
                ? __float2half(0.0f)
                : (__half2float(y[i]) > 0.0f)
                    ? dx[i]
                    : __hmul(leakSlope, dx[i]);
#else
            dy[i] = (__half2float(y[i]) > __half2float(clipping))
                ? __float2half(0.0f)
                : (__half2float(y[i]) > 0.0f)
                    ? dx[i]
                    : __float2half(__half2float(leakSlope)
                                   * __half2float(dx[i]));
#endif
        }
        else {
#if __CUDA_ARCH__ >= 530
            dy[i] = (__half2float(y[i]) > 0.0f) ? dx[i]
                                                : __hmul(leakSlope, dx[i]);
#else
            dy[i] = (__half2float(y[i]) > 0.0f) ? dx[i]
                : __float2half(__half2float(leakSlope) * __half2float(dx[i]));
#endif
        }
    }
}

// Saturation
template <class T>
__global__ void cudaSaturation_propagate_kernel(T* x,
                                                 T* y,
                                                 unsigned int size,
                                                 T threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        T value = x[i];

        if (threshold != 0.0f) {
            y[i] = (value < -threshold) ? -threshold
                 : (value > threshold) ? threshold
                 : value;
        }
    }
}

template <>
__global__ void cudaSaturation_propagate_kernel<__half>(__half* x,
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

template <class T>
__global__ void
cudaSaturation_backPropagate_kernel(T* y,
                                     T* dx,
                                     T* dy,
                                     unsigned int size,
                                     T threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (threshold != 0.0f) {
            dy[i] = dx[i] * ((y[i] > -threshold && y[i] < threshold)
                ? 1.0f : 0.0f);
        }
    }
}

template <>
__global__ void
cudaSaturation_backPropagate_kernel<__half>(__half* y,
                                     __half* dx,
                                     __half* dy,
                                     unsigned int size,
                                     __half threshold)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        if (__half2float(threshold) != 0.0f) {
#if __CUDA_ARCH__ >= 530
            dy[i] = (__hgt(y[i], __hneg(threshold)) && __hlt(y[i], threshold))
                ? dx[i] : __float2half(0.0f);
#else
            dy[i] = (__half2float(y[i]) > -__half2float(threshold)
                     && __half2float(y[i]) < __half2float(threshold))
                ? dx[i] : __float2half(0.0f);
#endif
        }
    }
}

// Softplus
template <class T>
__global__ void cudaSoftplus_propagate_kernel(T* x,
                                               T* y,
                                               unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = log(1.0f + exp(x[i]));
    }
}

template <>
__global__ void cudaSoftplus_propagate_kernel<__half>(__half* x,
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

template <class T>
__global__ void
cudaSoftplus_backPropagate_kernel(T* y, T* dx, T* dy, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dy[i] = dx[i] * (1.0f - exp(-y[i]));
    }
}

template <>
__global__ void
cudaSoftplus_backPropagate_kernel<__half>(__half* y, __half* dx, __half* dy, unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
// hexp and hlog are only available since CUDA 8.0
#if __CUDA_ARCH__ >= 530 && defined(CUDART_VERSION) && CUDART_VERSION >= 8000
        dy[i] = __hmul(dx[i], (__hsub(__float2half(1.0f), hexp(__hneg(y[i])))));
#else
        dy[i] = __float2half(__half2float(dx[i])
                             * (1.0f - exp(-__half2float(y[i]))));
#endif
    }
}

// Swish
template <class T>
__global__ void cudaSwish_propagate_kernel(T* x,
                                            T* y,
                                            T* sigmoid,
                                            unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        const T sig = 1.0f / (1.0f + exp(-x[i]));
        sigmoid[i] = sig;
        y[i] = x[i] * sig;
    }
}

template <>
__global__ void cudaSwish_propagate_kernel<__half>(__half* x,
                                            __half* y,
                                            __half* sigmoid,
                                            unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
// hexp and hlog are only available since CUDA 8.0
#if __CUDA_ARCH__ >= 530 && defined(CUDART_VERSION) && CUDART_VERSION >= 8000
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

template <class T>
__global__ void cudaSwish_backPropagate_kernel(T* y,
                                                T* dx,
                                                T* dy,
                                                T* sigmoid,
                                                unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        dy[i] = dx[i] * (sigmoid[i] + y[i] * (1.0f - sigmoid[i]));
    }
}

template <>
__global__ void cudaSwish_backPropagate_kernel<__half>(__half* y,
                                                __half* dx,
                                                __half* dy,
                                                __half* sigmoid,
                                                unsigned int size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
#if __CUDA_ARCH__ >= 530
        dy[i] = __hmul(dx[i], __hadd(sigmoid[i],
                             __hmul(y[i],
                                    __hsub(__float2half(1.0f), sigmoid[i]))));
#else
        const float sig = __half2float(sigmoid[i]);
        dy[i] = __float2half(__half2float(dx[i]) 
            * (sig + __half2float(y[i]) * (1.0f - sig)));
#endif
    }
}

namespace N2D2 {

// Rectifier
template <class T>
void cudaRectifier_propagate(T* x,
                             T* y,
                             unsigned int size,
                             T leakSlope,
                             T clipping)
{
    cudaRectifier_propagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(x),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
         size,
         reinterpret_cast<typename Cuda::cuda_type<T>::type&>(leakSlope),
         reinterpret_cast<typename Cuda::cuda_type<T>::type&>(clipping));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaRectifier_backPropagate(T* y,
                                 T* dx,
                                 T* dy,
                                 unsigned int size,
                                 T leakSlope,
                                 T clipping)
{
    cudaRectifier_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dx),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dy),
         size,
         reinterpret_cast<typename Cuda::cuda_type<T>::type&>(leakSlope),
         reinterpret_cast<typename Cuda::cuda_type<T>::type&>(clipping));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Saturation
template <class T>
void cudaSaturation_propagate(T* x,
                              T* y,
                              unsigned int size,
                              T threshold)
{
    cudaSaturation_propagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(x),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
         size,
         reinterpret_cast<typename Cuda::cuda_type<T>::type&>(threshold));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaSaturation_backPropagate(T* y,
                                  T* dx,
                                  T* dy,
                                  unsigned int size,
                                  T threshold)
{
    cudaSaturation_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dx),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dy),
         size,
         reinterpret_cast<typename Cuda::cuda_type<T>::type&>(threshold));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Softplus
template <class T>
void cudaSoftplus_propagate(T* x,
                            T* y,
                            unsigned int size)
{
    cudaSoftplus_propagate_kernel<<<(size + 255) / 256, 256>>>(
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(x),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
        size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaSoftplus_backPropagate(T* y,
                                T* dx,
                                T* dy,
                                unsigned int size)
{
    cudaSoftplus_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dx),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dy),
         size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

// Swish
template <class T>
void cudaSwish_propagate(T* x,
                         T* y,
                         T* sigmoid,
                         unsigned int size)
{
    cudaSwish_propagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(x),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(sigmoid),
         size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaSwish_backPropagate(T* y,
                             T* dx,
                             T* dy,
                             T* sigmoid,
                             unsigned int size)
{
    cudaSwish_backPropagate_kernel<<<(size + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dx),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dy),
         reinterpret_cast<typename Cuda::cuda_type<T>::type*>(sigmoid),
         size);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaRectifier_propagate(half_float::half* x,
                                    half_float::half* y,
                                    unsigned int size,
                                    half_float::half leakSlope,
                                    half_float::half clipping);
template void cudaRectifier_propagate(float* x,
                                    float* y,
                                    unsigned int size,
                                    float leakSlope,
                                    float clipping);
template void cudaRectifier_propagate(double* x,
                                    double* y,
                                    unsigned int size,
                                    double leakSlope,
                                    double clipping);

template void cudaRectifier_backPropagate(half_float::half* y,
                                        half_float::half* dx,
                                        half_float::half* dy,
                                        unsigned int size,
                                        half_float::half leakSlope,
                                        half_float::half clipping);
template void cudaRectifier_backPropagate(float* y,
                                        float* dx,
                                        float* dy,
                                        unsigned int size,
                                        float leakSlope,
                                        float clipping);
template void cudaRectifier_backPropagate(double* y,
                                        double* dx,
                                        double* dy,
                                        unsigned int size,
                                        double leakSlope,
                                        double clipping);

template void cudaSaturation_propagate(half_float::half* x,
                                     half_float::half* y,
                                     unsigned int size,
                                     half_float::half threshold);
template void cudaSaturation_propagate(float* x,
                                     float* y,
                                     unsigned int size,
                                     float threshold);
template void cudaSaturation_propagate(double* x,
                                     double* y,
                                     unsigned int size,
                                     double threshold);

template void cudaSaturation_backPropagate(half_float::half* y,
                                    half_float::half* dx,
                                    half_float::half* dy,
                                    unsigned int size,
                                    half_float::half threshold);
template void cudaSaturation_backPropagate(float* y,
                                    float* dx,
                                    float* dy,
                                    unsigned int size,
                                    float threshold);
template void cudaSaturation_backPropagate(double* y,
                                    double* dx,
                                    double* dy,
                                    unsigned int size,
                                    double threshold);

template void cudaSoftplus_propagate(half_float::half* x, half_float::half* y, unsigned int size);
template void cudaSoftplus_propagate(float* x, float* y, unsigned int size);
template void cudaSoftplus_propagate(double* x, double* y, unsigned int size);

template void cudaSoftplus_backPropagate(half_float::half* y, half_float::half* dx, half_float::half* dy, unsigned int size);
template void cudaSoftplus_backPropagate(float* y, float* dx, float* dy, unsigned int size);
template void cudaSoftplus_backPropagate(double* y, double* dx, double* dy, unsigned int size);

template void cudaSwish_propagate(half_float::half* x,
                               half_float::half* y,
                               half_float::half* sigmoid,
                               unsigned int size);
template void cudaSwish_propagate(float* x,
                               float* y,
                               float* sigmoid,
                               unsigned int size);
template void cudaSwish_propagate(double* x,
                               double* y,
                               double* sigmoid,
                               unsigned int size);

template void cudaSwish_backPropagate(half_float::half* y,
                                   half_float::half* dx,
                                   half_float::half* dy,
                                   half_float::half* sigmoid,
                                   unsigned int size);
template void cudaSwish_backPropagate(float* y,
                                   float* dx,
                                   float* dy,
                                   float* sigmoid,
                                   unsigned int size);
template void cudaSwish_backPropagate(double* y,
                                   double* dx,
                                   double* dy,
                                   double* sigmoid,
                                   unsigned int size);

}
