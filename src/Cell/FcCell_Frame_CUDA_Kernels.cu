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

#include "Cell/FcCell_Frame_CUDA_Kernels.hpp"

#if __CUDA_ARCH__ < 600
// Prototype declaration
__device__ __forceinline__ double atomicAdd(double* address, double val);
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template <class T>
__global__
void cudaFcWeightsSumSq_kernel(T* weights,
                                    T* weightsNorm,
                                    unsigned int nbChannels,
                                    unsigned int nbOutputs)
{
    __shared__ T sumSq[256];

    // each thread loads one element from global to shared mem
    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nbChannels) {
        sumSq[tid] = weights[index + nbChannels * blockIdx.z]
                        * weights[index + nbChannels * blockIdx.z];
    }
    else
        sumSq[tid] = 0.0f;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride>>= 1) {
        if (tid < stride) {
            sumSq[tid] += sumSq[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        atomicAdd(weightsNorm + blockIdx.z, sumSq[0]);
}

template <class T>
__global__
void cudaFcWeightsSqrt_kernel(T* weightsNorm,
                                unsigned int nbOutputs,
                                T epsilon)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < nbOutputs; i += stride)
        weightsNorm[i] = sqrtf(weightsNorm[i] + epsilon);
}

template <class T>
__global__
void cudaFcWeightsNormalize_kernel(T* weights,
                                    T* weightsNorm,
                                    unsigned int nbChannels,
                                    unsigned int nbOutputs)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < nbChannels; i += stride)
        weights[i + nbChannels * blockIdx.z] /= weightsNorm[blockIdx.z];
}

namespace N2D2 {

template <class T>
void cudaFcWeightsSumSq(const cudaDeviceProp& deviceProp,
                               T* weights,
                               T* weightsNorm,
                               unsigned int nbChannels,
                               unsigned int nbOutputs)
{
    const dim3 blocksPerGrid = {(nbChannels + 255) / 256, 1, nbOutputs};
    const dim3 threadsPerBlocks = {256, 1, 1};

    cudaFcWeightsSumSq_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(weights),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(weightsNorm),
           nbChannels,
           nbOutputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <>
void cudaFcWeightsSumSq<half_float::half>(const cudaDeviceProp& deviceProp,
                               half_float::half* weights,
                               half_float::half* weightsNorm,
                               unsigned int nbChannels,
                               unsigned int nbOutputs)
{
    //TODO: when implementing the CUDA kernel,
    // this specialization should be removed.
    throw std::runtime_error("cudaFcWeightsSumSq<half_float::half>"
        " not implemented!");
}

template <class T>
void cudaFcWeightsSqrt(const cudaDeviceProp& deviceProp,
                              T* weightsNorm,
                              unsigned int nbOutputs,
                              T epsilon)
{
    cudaFcWeightsSqrt_kernel<<<(nbOutputs + 255) / 256, 256>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(weightsNorm),
           nbOutputs,
           reinterpret_cast<typename Cuda::cuda_type<T>::type&>(epsilon));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <>
void cudaFcWeightsSqrt<half_float::half>(const cudaDeviceProp& deviceProp,
                              half_float::half* weightsNorm,
                              unsigned int nbOutputs,
                              half_float::half epsilon)
{
    //TODO: when implementing the CUDA kernel,
    // this specialization should be removed.
    throw std::runtime_error("cudaFcWeightsSqrt<half_float::half>"
        " not implemented!");
}

template <class T>
void cudaFcWeightsNormalize(const cudaDeviceProp& deviceProp,
                                   T* weights,
                                   T* weightsNorm,
                                   unsigned int nbChannels,
                                   unsigned int nbOutputs)
{
    const dim3 blocksPerGrid = {(nbChannels + 255) / 256, 1, nbOutputs};
    const dim3 threadsPerBlocks = {256, 1, 1};

    cudaFcWeightsNormalize_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type*>(weights),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(weightsNorm),
           nbChannels,
           nbOutputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <>
void cudaFcWeightsNormalize<half_float::half>(const cudaDeviceProp& deviceProp,
                                   half_float::half* weights,
                                   half_float::half* weightsNorm,
                                   unsigned int nbChannels,
                                   unsigned int nbOutputs)
{
    //TODO: when implementing the CUDA kernel,
    // this specialization should be removed.
    throw std::runtime_error("cudaFcWeightsNormalize<half_float::half>"
        " not implemented!");
}


template void cudaFcWeightsSumSq(const cudaDeviceProp& deviceProp,
                               half_float::half* weights,
                               half_float::half* weightsNorm,
                               unsigned int nbChannels,
                               unsigned int nbOutputs);
template void cudaFcWeightsSumSq(const cudaDeviceProp& deviceProp,
                               float* weights,
                               float* weightsNorm,
                               unsigned int nbChannels,
                               unsigned int nbOutputs);
template void cudaFcWeightsSumSq(const cudaDeviceProp& deviceProp,
                               double* weights,
                               double* weightsNorm,
                               unsigned int nbChannels,
                               unsigned int nbOutputs);

template void cudaFcWeightsSqrt(const cudaDeviceProp& deviceProp,
                              half_float::half* weightsNorm,
                              unsigned int nbOutputs,
                              half_float::half epsilon);
template void cudaFcWeightsSqrt(const cudaDeviceProp& deviceProp,
                              float* weightsNorm,
                              unsigned int nbOutputs,
                              float epsilon);
template void cudaFcWeightsSqrt(const cudaDeviceProp& deviceProp,
                              double* weightsNorm,
                              unsigned int nbOutputs,
                              double epsilon);

template void cudaFcWeightsNormalize(const cudaDeviceProp& deviceProp,
                                   half_float::half* weights,
                                   half_float::half* weightsNorm,
                                   unsigned int nbChannels,
                                   unsigned int nbOutputs);
template void cudaFcWeightsNormalize(const cudaDeviceProp& deviceProp,
                                   float* weights,
                                   float* weightsNorm,
                                   unsigned int nbChannels,
                                   unsigned int nbOutputs);
template void cudaFcWeightsNormalize(const cudaDeviceProp& deviceProp,
                                   double* weights,
                                   double* weightsNorm,
                                   unsigned int nbChannels,
                                   unsigned int nbOutputs);

}
