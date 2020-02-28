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

//Float
__global__
void cudaSFcWeightsSumSq_kernel(float* weights,
                                    float* weightsNorm,
                                    unsigned int nbChannels,
                                    unsigned int nbOutputs)
{
    __shared__ float sumSq[256];

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

__global__
void cudaSFcWeightsSqrt_kernel(float* weightsNorm,
                                unsigned int nbOutputs,
                                float epsilon)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < nbOutputs; i += stride)
        weightsNorm[i] = sqrtf(weightsNorm[i] + epsilon);
}

__global__
void cudaSFcWeightsNormalize_kernel(float* weights,
                                    float* weightsNorm,
                                    unsigned int nbChannels,
                                    unsigned int nbOutputs)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < nbChannels; i += stride)
        weights[i + nbChannels * blockIdx.z] /= weightsNorm[blockIdx.z];
}

//Double
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

__global__
void cudaDFcWeightsSumSq_kernel(double* weights,
                                    double* weightsNorm,
                                    unsigned int nbChannels,
                                    unsigned int nbOutputs)
{
    __shared__ double sumSq[256];

    // each thread loads one element from global to shared mem
    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nbChannels) {
        sumSq[tid] = weights[index + nbChannels * blockIdx.z]
                        * weights[index + nbChannels * blockIdx.z];
    }
    else
        sumSq[tid] = 0.0;

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

__global__
void cudaDFcWeightsSqrt_kernel(double* weightsNorm,
                                unsigned int nbOutputs,
                                double epsilon)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < nbOutputs; i += stride)
        weightsNorm[i] = sqrt(weightsNorm[i] + epsilon);
}

__global__
void cudaDFcWeightsNormalize_kernel(double* weights,
                                    double* weightsNorm,
                                    unsigned int nbChannels,
                                    unsigned int nbOutputs)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < nbChannels; i += stride)
        weights[i + nbChannels * blockIdx.z] /= weightsNorm[blockIdx.z];
}

//Float
void N2D2::cudaSFcWeightsSumSq(const cudaDeviceProp& deviceProp,
                               float* weights,
                               float* weightsNorm,
                               unsigned int nbChannels,
                               unsigned int nbOutputs)
{
    const dim3 blocksPerGrid = {(nbChannels + 255) / 256, 1, nbOutputs};
    const dim3 threadsPerBlocks = {256, 1, 1};

    cudaSFcWeightsSumSq_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (weights,
           weightsNorm,
           nbChannels,
           nbOutputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSFcWeightsSqrt(const cudaDeviceProp& deviceProp,
                              float* weightsNorm,
                              unsigned int nbOutputs,
                              float epsilon)
{
    cudaSFcWeightsSqrt_kernel<<<(nbOutputs + 255) / 256, 256>>>
        (weightsNorm,
           nbOutputs,
           epsilon);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSFcWeightsNormalize(const cudaDeviceProp& deviceProp,
                                   float* weights,
                                   float* weightsNorm,
                                   unsigned int nbChannels,
                                   unsigned int nbOutputs)
{
    const dim3 blocksPerGrid = {(nbChannels + 255) / 256, 1, nbOutputs};
    const dim3 threadsPerBlocks = {256, 1, 1};

    cudaSFcWeightsNormalize_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (weights,
           weightsNorm,
           nbChannels,
           nbOutputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

//Double
void N2D2::cudaDFcWeightsSumSq(const cudaDeviceProp& deviceProp,
                               double* weights,
                               double* weightsNorm,
                               unsigned int nbChannels,
                               unsigned int nbOutputs)
{
    const dim3 blocksPerGrid = {(nbChannels + 255) / 256, 1, nbOutputs};
    const dim3 threadsPerBlocks = {256, 1, 1};

    cudaDFcWeightsSumSq_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (weights,
           weightsNorm,
           nbChannels,
           nbOutputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDFcWeightsSqrt(const cudaDeviceProp& deviceProp,
                              double* weightsNorm,
                              unsigned int nbOutputs,
                              double epsilon)
{
    cudaDFcWeightsSqrt_kernel<<<(nbOutputs + 255) / 256, 256>>>
        (weightsNorm,
           nbOutputs,
           epsilon);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaDFcWeightsNormalize(const cudaDeviceProp& deviceProp,
                                   double* weights,
                                   double* weightsNorm,
                                   unsigned int nbChannels,
                                   unsigned int nbOutputs)
{
    const dim3 blocksPerGrid = {(nbChannels + 255) / 256, 1, nbOutputs};
    const dim3 threadsPerBlocks = {256, 1, 1};

    cudaDFcWeightsNormalize_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (weights,
           weightsNorm,
           nbChannels,
           nbOutputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
