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

#include "Cell/NormalizeCell_Frame_CUDA_Kernels.hpp"

////Forward
template <class T>
__global__
void cudaNormalizeL2Forward_kernel(const T alpha,
                                    T* inputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    unsigned int batchSize,
                                    const T beta,
                                    T* outputs,
                                    T* normData,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth)
{
    const unsigned int batchOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int oy = threadIdx.y; oy < outputsHeight;
            oy += blockDim.y) {
        for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                ox += blockDim.x)
        {
            T sumSq = 0.0f;

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                sumSq += inputs[idx] * inputs[idx];
            }

            const T scale = sqrt(sumSq + 1.0e-6);

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                normData[idx] = scale;
                outputs[idx] = inputs[idx] / scale;
            }
        }
    }
}

template <>
__global__
void cudaNormalizeL2Forward_kernel<__half>(const __half alpha,
                                    __half* inputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    unsigned int batchSize,
                                    const __half beta,
                                    __half* outputs,
                                    __half* normData,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth)
{
    const unsigned int batchOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int oy = threadIdx.y; oy < outputsHeight;
            oy += blockDim.y) {
        for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                ox += blockDim.x)
        {
#if __CUDA_ARCH__ >= 530
            __half sumSq = __float2half(0.0f);

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                sumSq = __hadd(sumSq, __hmul(inputs[idx], inputs[idx]));
            }

            const __half scale = __float2half(
                sqrt(__half2float(sumSq) + 1.0e-6));

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                normData[idx] = scale;
                outputs[idx] = __float2half(__half2float(inputs[idx])
                                    / __half2float(scale));
            }
#else
            float sumSq(0.0f);

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                sumSq += __half2float(inputs[idx]) * __half2float(inputs[idx]);
            }

            const float scale = sqrt(sumSq + 1.0e-6);

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                normData[idx] = __float2half(scale);
                outputs[idx] = __float2half(__half2float(inputs[idx]) / scale);
            }
#endif
        }
    }
}

// Backward
template <class T>
__global__
void cudaNormalizeL2Backward_kernel(const T alpha,
                                     T* outputs,
                                     T* normData,
                                     T* diffInputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     unsigned int batchSize,
                                     const T beta,
                                     T* diffOutputs,
                                     unsigned int nbChannels,
                                     unsigned int channelsHeight,
                                     unsigned int channelsWidth)
{
    const unsigned int batchOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int oy = threadIdx.y; oy < outputsHeight;
            oy += blockDim.y) {
        for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                ox += blockDim.x)
        {
            T a = 0.0f;

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                a += outputs[idx] * diffInputs[idx];
            }

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                if (beta != 0.0f) {
                    diffOutputs[idx] = (diffInputs[idx] - outputs[idx] * a)
                        / normData[idx] + beta * diffOutputs[idx];
                }
                else {
                    diffOutputs[idx] = (diffInputs[idx] - outputs[idx] * a)
                        / normData[idx];
                }
            }
        }
    }
}

template <>
__global__
void cudaNormalizeL2Backward_kernel<__half>(const __half alpha,
                                     __half* outputs,
                                     __half* normData,
                                     __half* diffInputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     unsigned int batchSize,
                                     const __half beta,
                                     __half* diffOutputs,
                                     unsigned int nbChannels,
                                     unsigned int channelsHeight,
                                     unsigned int channelsWidth)
{
    const unsigned int batchOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int oy = threadIdx.y; oy < outputsHeight;
            oy += blockDim.y) {
        for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                ox += blockDim.x)
        {
#if __CUDA_ARCH__ >= 530
            __half a = __float2half(0.0f);

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                a = __hadd(a, __hmul(outputs[idx], diffInputs[idx]));
            }

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                const __half factor = __float2half(1.0f
                                            / __half2float(normData[idx]));

                if (! __heq(beta, __float2half(0.0f))) {
                    diffOutputs[idx] = __hadd(
                        __hmul(__hsub(diffInputs[idx], __hmul(outputs[idx], a)),
                              factor),
                        __hmul(beta, diffOutputs[idx]));
                }
                else {
                    diffOutputs[idx] =
                        __hmul(__hsub(diffInputs[idx], __hmul(outputs[idx], a)),
                              factor);
                }
            }
#else
            float a = 0.0f;

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                a += __half2float(outputs[idx]) * __half2float(diffInputs[idx]);
            }

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                const unsigned int idx = batchOffset
                    + ox + (oy + output * outputsHeight) * outputsWidth;

                if (__half2float(beta) != 0.0f) {
                    diffOutputs[idx] = __float2half(
                        (__half2float(diffInputs[idx])
                            - __half2float(outputs[idx]) * a)
                                / __half2float(normData[idx])
                        + __half2float(beta) * __half2float(diffOutputs[idx]));
                }
                else {
                    diffOutputs[idx] = __float2half(
                        (__half2float(diffInputs[idx])
                            - __half2float(outputs[idx]) * a)
                                / __half2float(normData[idx]));
                }
            }
#endif
        }
    }
}

namespace N2D2 {

template <class T>
void cudaNormalizeL2Forward(const cudaDeviceProp& deviceProp,
                                   T alpha,
                                   T* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   T beta,
                                   T* outputs,
                                   T* normData,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;

    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);
    const dim3 blocksPerGrid = {1, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaNormalizeL2Forward_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type&>(alpha),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(inputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           reinterpret_cast<typename Cuda::cuda_type<T>::type&>(beta),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(normData),
           nbOutputs,
           outputsHeight,
           outputsWidth);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaNormalizeL2Backward(const cudaDeviceProp& deviceProp,
                                    T alpha,
                                    T* outputs,
                                    T* normData,
                                    T* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    T beta,
                                    T* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {1, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaNormalizeL2Backward_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type&>(alpha),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(normData),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffInputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           reinterpret_cast<typename Cuda::cuda_type<T>::type&>(beta),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffOutputs),
           nbChannels,
           channelsHeight,
           channelsWidth);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaNormalizeL2Forward(const cudaDeviceProp& deviceProp,
                                   half_float::half alpha,
                                   half_float::half* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   half_float::half beta,
                                   half_float::half* outputs,
                                   half_float::half* normData,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth);
template void cudaNormalizeL2Forward(const cudaDeviceProp& deviceProp,
                                   float alpha,
                                   float* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   float beta,
                                   float* outputs,
                                   float* normData,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth);
template void cudaNormalizeL2Forward(const cudaDeviceProp& deviceProp,
                                   double alpha,
                                   double* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   double beta,
                                   double* outputs,
                                   double* normData,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth);

template void cudaNormalizeL2Backward(const cudaDeviceProp& deviceProp,
                                    half_float::half alpha,
                                    half_float::half* outputs,
                                    half_float::half* normData,
                                    half_float::half* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    half_float::half beta,
                                    half_float::half* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth);
template void cudaNormalizeL2Backward(const cudaDeviceProp& deviceProp,
                                    float alpha,
                                    float* outputs,
                                    float* normData,
                                    float* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    float beta,
                                    float* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth);
template void cudaNormalizeL2Backward(const cudaDeviceProp& deviceProp,
                                    double alpha,
                                    double* outputs,
                                    double* normData,
                                    double* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    double beta,
                                    double* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth);

}
