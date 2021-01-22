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

template <class T>
__global__ void cudaZeroInit_kernel(unsigned int size,
                                     T* data)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        data[i] = T(0);
}

template <class T>
__global__ void cudaSqrt_kernel(unsigned int size,
                                 T* data)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < size; i += stride)
        data[i] = sqrt(data[i]);
}

template <class T>
__global__ void cudaMult_kernel(unsigned int size,
                                 T* a,
                                 T* b,
                                 const T beta,
                                 T* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (beta != 0.0f) {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = a[i] * b[i] + beta * result[i];
    }
    else {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = a[i] * b[i];
    }
}

template <class T>
__global__ void cudaScale_kernel(unsigned int size,
                                  T* input,
                                  const T scale,
                                  const T shift,
                                  const T beta,
                                  T* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (beta != 0.0f) {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = input[i] * scale + shift + beta * result[i];
    }
    else {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = input[i] * scale  + shift;
    }
}

template <class T>
__global__ void cudaScaleAbs_kernel(unsigned int size,
                                     T* input,
                                     const T scale,
                                     const T beta,
                                     T* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (beta != 0.0f) {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = fabs(input[i]) * scale + beta * result[i];
    }
    else {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = fabs(input[i]) * scale;
    }
}

template <class T>
__global__ void cudaScaleSign_kernel(unsigned int size,
                                      T* input,
                                      T* sign,
                                      const T scale,
                                      const T beta,
                                      T* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (beta != 0.0f) {
        for (unsigned int i = index; i < size; i += stride) {
            const float sgn = (sign[i] >= 0) ? 1.0f : -1.0f;
            result[i] = input[i] * sgn * scale + beta * result[i];
        }
    }
    else {
        for (unsigned int i = index; i < size; i += stride) {
            const float sgn = (sign[i] >= 0) ? 1.0f : -1.0f;
            result[i] = input[i] * sgn * scale;
        }
    }
}

template <class T>
__global__ void cudaScaleSquare_kernel(unsigned int size,
                                        T* input,
                                        const T scale,
                                        const T shift,
                                        const T beta,
                                        T* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (beta != 0.0f) {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = input[i] * input[i] * scale 
                            + shift + beta * result[i];
    }
    else {
        for (unsigned int i = index; i < size; i += stride)
            result[i] = input[i] * input[i] * scale
                            + shift;
    }
}

template <class T>
__global__ void cudaMaxForward_kernel(unsigned int size,
                                       T* input,
                                       T* maxVal,
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

template <class T>
__global__ void cudaMaxBackward_kernel(unsigned int size,
                                        T* diffInput,
                                        const unsigned int idx,
                                        unsigned int* argMax,
                                        const T beta,
                                        T* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (beta != 0.0f) {
        for (unsigned int i = index; i < size; i += stride) {
            result[i] = (argMax[i] == idx) ? (diffInput[i] + beta * result[i])
                                           : beta * result[i];
        }
    }
    else {
        for (unsigned int i = index; i < size; i += stride) {
            result[i] = (argMax[i] == idx) ? diffInput[i]
                                           : 0.0f;
        }
    }
}

template <class T>
__global__ void cudaEuclideanSumBackward_kernel(unsigned int size,
                                                 T* diffInput,
                                                 T* input,
                                                 T* output,
                                                 const T scale,
                                                 const T beta,
                                                 T* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    if (beta != 0.0f) {
        for (unsigned int i = index; i < size; i += stride) {
            result[i] = (output[i] != 0.0f)
                ? diffInput[i] * scale * (input[i] / output[i]) + beta * result[i]
                : beta * result[i];
        }
    }
    else {
        for (unsigned int i = index; i < size; i += stride) {
            result[i] = (output[i] != 0.0f)
                ? diffInput[i] * scale * (input[i] / output[i])
                : 0.0f;
        }
    }
}

namespace N2D2 {

template <class T>
void cudaZeroInit(unsigned int size,
                         T* data)
{
    cudaZeroInit_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(data));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaSqrt(unsigned int size,
                     T* data)
{
    cudaSqrt_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(data));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaMult(unsigned int size,
                     T* a,
                     T* b,
                     const T beta,
                     T* result)
{
    cudaMult_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(a),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(b),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(beta),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(result));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaScale(unsigned int size,
                      T* input,
                      const T scale,
                      const T shift,
                      const T beta,
                      T* result)
{
    cudaScale_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(input),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(scale),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(shift),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(beta),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(result));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaScaleAbs(unsigned int size,
                         T* input,
                         const T scale,
                         const T beta,
                         T* result)
{
    cudaScaleAbs_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(input),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(scale),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(beta),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(result));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaScaleSign(unsigned int size,
                          T* input,
                          T* sign,
                          const T scale,
                          const T beta,
                          T* result)
{
    cudaScaleSign_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(input),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(sign),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(scale),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(beta),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(result));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaScaleSquare(unsigned int size,
                            T* input,
                            const T scale,
                            const T shift,
                            const T beta,
                            T* result)
{
    cudaScaleSquare_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(input),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(scale),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(shift),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(beta),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(result));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaMaxForward(unsigned int size,
                           T* input,
                           T* maxVal,
                           const unsigned int idx,
                           unsigned int* argMax)
{
    cudaMaxForward_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(input),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(maxVal),
        idx,
        argMax);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaMaxBackward(unsigned int size,
                            T* diffInput,
                            const unsigned int idx,
                            unsigned int* argMax,
                            const T beta,
                            T* result)
{
    cudaMaxBackward_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffInput),
        idx,
        argMax,
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(beta),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(result));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaEuclideanSumBackward(unsigned int size,
                                     T* diffInput,
                                     T* input,
                                     T* output,
                                     const T scale,
                                     const T beta,
                                     T* result)
{
    cudaEuclideanSumBackward_kernel<<<(size + 255) / 256, 256>>>(size,
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffInput),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(input),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(output),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(scale),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(beta),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(result));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaZeroInit(unsigned int size, unsigned int* data);
template void cudaZeroInit(unsigned int size, float* data);

template void cudaSqrt(unsigned int size, float* data);

template void cudaMult(unsigned int size,
    float* a,
    float* b,
    const float beta,
    float* result);

template void cudaScale(unsigned int size,
    float* input,
    const float scale,
    const float shift,
    const float beta,
    float* result);

template void cudaScaleAbs(unsigned int size,
    float* input,
    const float scale,
    const float beta,
    float* result);

template void cudaScaleSign(unsigned int size,
                          float* input,
                          float* sign,
                          const float scale,
                          const float beta,
                          float* result);

template void cudaScaleSquare(unsigned int size,
                            float* input,
                            const float scale,
                            const float shift,
                            const float beta,
                            float* result);

template void cudaMaxForward(unsigned int size,
                           float* input,
                           float* maxVal,
                           const unsigned int idx,
                           unsigned int* argMax);

template void cudaMaxBackward(unsigned int size,
                            float* diffInput,
                            const unsigned int idx,
                            unsigned int* argMax,
                            const float beta,
                            float* result);

template void cudaEuclideanSumBackward(unsigned int size,
                                     float* diffInput,
                                     float* input,
                                     float* output,
                                     const float scale,
                                     const float beta,
                                     float* result);

}
