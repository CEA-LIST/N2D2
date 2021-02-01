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

template <class T>
__global__ void
cudaClamp_kernel(T* x, unsigned int size, T minVal, T maxVal)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        x[i] = (x[i] < minVal) ? minVal :
               (x[i] > maxVal) ? maxVal :
                                 x[i];
    }
}

template <>
__global__ void
cudaClamp_kernel<__half>(__half* x, unsigned int size, __half minVal, __half maxVal)
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

template <class T>
__global__ void cudaPow_kernel(unsigned int size,
                                 T power,
                                 const T *x,
                                 T *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = powf(x[i], power);
    }
}

template <>
__global__ void cudaPow_kernel<__half>(unsigned int size,
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

template <class T>
__global__ void cudaAdd_kernel(unsigned int size,
                                 T value,
                                 const T *x,
                                 T *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = x[i] + value;
    }
}

template <>
__global__ void cudaAdd_kernel<__half>(unsigned int size,
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

template <class T>
__global__ void cudaMult_kernel(unsigned int size,
                                 const T *x1,
                                 const T *x2,
                                 T *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = x1[i] * x2[i];
    }
}

template <>
__global__ void cudaMult_kernel<__half>(unsigned int size,
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

template <class T>
__global__ void cudaInv_kernel(unsigned int size,
                                 const T *x,
                                 T *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = 1.0f / x[i];
    }
}

template <>
__global__ void cudaInv_kernel<__half>(unsigned int size,
                                 const __half *x,
                                 __half *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride) {
        y[i] = __float2half(1.0f / __half2float(x[i]));
    }
}


namespace N2D2 {

template <class T>
void cudaClamp(T* x, unsigned int size, T minVal, T maxVal)
{
    cudaClamp_kernel<<<(size + 255) / 256, 256>>>(reinterpret_cast<typename Cuda::cuda_type<T>::type*>(x),
                                            size,
                                            reinterpret_cast<typename Cuda::cuda_type<T>::type&>(minVal),
                                            reinterpret_cast<typename Cuda::cuda_type<T>::type&>(maxVal));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
std::pair<T, T> cudaMinMax(T* x, unsigned int size)
{
    // Compute global min & max value on the full tensor
    thrust::device_ptr<T> thrustPtr(x);
    thrust::pair<typename thrust::device_vector<T>::iterator,
                 typename thrust::device_vector<T>::iterator> minMaxPair
        = thrust::minmax_element(thrustPtr, thrustPtr + size);

    return std::make_pair(*(minMaxPair.first), *(minMaxPair.second));
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

template <>
std::pair<half_float::half, half_float::half>
cudaMinMax(half_float::half* x, unsigned int size)
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

template <class T>
void cudaPow(unsigned int size,
                      T power,
                      const T *x,
                      T *y)
{
    cudaPow_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type&>(power),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(x),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaAdd(unsigned int size,
                    T value,
                    const T *x,
                    T *y)
{
    cudaAdd_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type&>(value),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(x),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaMult(unsigned int size,
                      const T *x1,
                      const T *x2,
                      T *y)
{
    cudaMult_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(x1),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(x2),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaInv(unsigned int size,
                    const T *x,
                    T *y)
{
    cudaInv_kernel<<<(size + 255) / 256, 256>>>
        (size,
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(x),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(y));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaClamp(half_float::half* x, unsigned int size, half_float::half minVal, half_float::half maxVal);
template void cudaClamp(float* x, unsigned int size, float minVal, float maxVal);
template void cudaClamp(double* x, unsigned int size, double minVal, double maxVal);

template std::pair<half_float::half, half_float::half> cudaMinMax(half_float::half* x, unsigned int size);
template std::pair<float, float> cudaMinMax(float* x, unsigned int size);
template std::pair<double, double> cudaMinMax(double* x, unsigned int size);

template void cudaPow(unsigned int size,
                      half_float::half power,
                      const half_float::half *x,
                      half_float::half *y);
template void cudaPow(unsigned int size,
                      float power,
                      const float *x,
                      float *y);
template void cudaPow(unsigned int size,
                      double power,
                      const double *x,
                      double *y);

template void cudaAdd(unsigned int size,
                    half_float::half value,
                    const half_float::half *x,
                    half_float::half *y);
template void cudaAdd(unsigned int size,
                    float value,
                    const float *x,
                    float *y);
template void cudaAdd(unsigned int size,
                    double value,
                    const double *x,
                    double *y);

template void cudaMult(unsigned int size,
                      const half_float::half *x1,
                      const half_float::half *x2,
                      half_float::half *y);
template void cudaMult(unsigned int size,
                      const float *x1,
                      const float *x2,
                      float *y);
template void cudaMult(unsigned int size,
                      const double *x1,
                      const double *x2,
                      double *y);

template void cudaInv(unsigned int size,
                    const half_float::half *x,
                    half_float::half *y);
template void cudaInv(unsigned int size,
                    const float *x,
                    float *y);
template void cudaInv(unsigned int size,
                    const double *x,
                    double *y);

}
