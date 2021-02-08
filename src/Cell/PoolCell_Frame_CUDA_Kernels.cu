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

#include "Cell/PoolCell_Frame_CUDA_Kernels.hpp"

////Forward Average
template <class T>
__global__
void cudaPoolForwardAverage_kernel(const T alpha,
                                    T* inputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    const T beta,
                                    T* outputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    bool countIncludePadding,
                                    char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)max(
                    desc->padding[0] - (int)(ox * desc->stride[0]), 0);
                const unsigned int syMin = (unsigned int)max(
                    desc->padding[1] - (int)(oy * desc->stride[1]), 0);
                const unsigned int sxMax = min(
                    max(channelsWidth + desc->padding[0]
                                  - ox * desc->stride[0], 0),
                    desc->pool[0]);
                const unsigned int syMax = min(
                    max(channelsHeight + desc->padding[1]
                                  - oy * desc->stride[1], 0),
                    desc->pool[1]);

                const int ix = (int)(ox * desc->stride[0]) - desc->padding[0];
                const int iy = (int)(oy * desc->stride[1]) - desc->padding[1];

                // For each output, compute the pool value
                T poolValue = 0.0f;
                unsigned int poolCount = 0;

                for (unsigned int channel = 0; channel < nbChannels;
                     ++channel)
                {
                    if (maps != NULL && !maps[output + channel * nbOutputs])
                        continue;

                    for (unsigned int sy = syMin; sy < syMax; ++sy) {
                        for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                            const unsigned int inputsIdx
                                = (ix + sx)
                                    + ((iy + sy) + channel * channelsHeight)
                                        * channelsWidth;

                            poolValue += inputs[inputsIdx + batchInputOffset];
                        }
                    }

                    poolCount += (countIncludePadding)
                        ? (desc->pool[0] * desc->pool[1])
                        : (sxMax - sxMin)*(syMax - syMin);
                }

                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth;

                if (beta != 0.0f) {
                    outputs[outputsIdx + batchOutputOffset]
                        = alpha * ((poolCount > 0) ?
                                      (poolValue / poolCount) : 0.0)
                          + beta * outputs[outputsIdx + batchOutputOffset];
                }
                else {
                    outputs[outputsIdx + batchOutputOffset]
                        = alpha * ((poolCount > 0) ?
                                      (poolValue / poolCount) : 0.0);
                }
            }
        }
    }
}

template <>
__global__
void cudaPoolForwardAverage_kernel<__half>(const __half alpha,
                                    __half* inputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    const __half beta,
                                    __half* outputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    bool countIncludePadding,
                                    char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)max(
                    desc->padding[0] - (int)(ox * desc->stride[0]), 0);
                const unsigned int syMin = (unsigned int)max(
                    desc->padding[1] - (int)(oy * desc->stride[1]), 0);
                const unsigned int sxMax = min(
                    max(channelsWidth + desc->padding[0]
                                  - ox * desc->stride[0], 0),
                    desc->pool[0]);
                const unsigned int syMax = min(
                    max(channelsHeight + desc->padding[1]
                                  - oy * desc->stride[1], 0),
                    desc->pool[1]);

                const int ix = (int)(ox * desc->stride[0]) - desc->padding[0];
                const int iy = (int)(oy * desc->stride[1]) - desc->padding[1];

                // For each output, compute the pool value
#if __CUDA_ARCH__ >= 530
                __half poolValue = __float2half(0.0f);
#else
                float poolValue = 0.0f;
#endif
                unsigned int poolCount = 0;

                for (unsigned int channel = 0; channel < nbChannels;
                     ++channel)
                {
                    if (maps != NULL && !maps[output + channel * nbOutputs])
                        continue;

                    for (unsigned int sy = syMin; sy < syMax; ++sy) {
                        for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                            const unsigned int inputsIdx
                                = (ix + sx)
                                    + ((iy + sy) + channel * channelsHeight)
                                        * channelsWidth;
#if __CUDA_ARCH__ >= 530
                            poolValue = __hadd(poolValue,
                                          inputs[inputsIdx + batchInputOffset]);
#else
                            poolValue += __half2float(inputs[inputsIdx
                                                           + batchInputOffset]);
#endif
                        }
                    }
                    poolCount += (countIncludePadding)
                        ? (desc->pool[0] * desc->pool[1])
                        : (sxMax - sxMin)*(syMax - syMin);
                }

                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth;

#if __CUDA_ARCH__ >= 530
                __half finalOutput = __float2half(0.0f);

                if (poolCount > 0) {
                    finalOutput = __hmul(__hmul(poolValue, alpha),
                                         __float2half(1.0f / poolCount));
                }

                if (!__heq(beta, __float2half(0.0f))) {
                    finalOutput = __hadd(finalOutput,
                        __hmul(outputs[outputsIdx + batchOutputOffset], beta));
                }

                outputs[outputsIdx + batchOutputOffset] = finalOutput;
#else
                if (__half2float(beta) != 0.0f) {
                    outputs[outputsIdx + batchOutputOffset]
                        = __float2half( __half2float(alpha) * ((poolCount > 0)
                                            ? (poolValue / poolCount) : 0.0f)
                          + __half2float(beta) * __half2float(outputs[outputsIdx
                                                        + batchOutputOffset]));
                }
                else {
                    outputs[outputsIdx + batchOutputOffset]
                        = __float2half( __half2float(alpha) * ((poolCount > 0)
                                            ? (poolValue / poolCount) : 0.0f));
                }
#endif
            }
        }
    }
}

////Forward MAX
template <class T>
__global__
void cudaPoolForwardMax_kernel(const T alpha,
                                T* inputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                const T beta,
                                T* outputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                bool useArgMax,
                                char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)max(
                    desc->padding[0] - (int)(ox * desc->stride[0]), 0);
                const unsigned int syMin = (unsigned int)max(
                    desc->padding[1] - (int)(oy * desc->stride[1]), 0);
                const unsigned int sxMax = min(
                    max(channelsWidth + desc->padding[0]
                                  - ox * desc->stride[0], 0),
                    desc->pool[0]);
                const unsigned int syMax = min(
                    max(channelsHeight + desc->padding[1]
                                  - oy * desc->stride[1], 0),
                    desc->pool[1]);

                const int ix = (int)(ox * desc->stride[0]) - desc->padding[0];
                const int iy = (int)(oy * desc->stride[1]) - desc->padding[1];

                T poolValue = 0.0f;

                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;

                // For each output, compute the pool value
                if (useArgMax) {
                    const N2D2::PoolCell_Frame_Kernels::ArgMax inputMax
                        = argMax[outputsIdx];

                    if (inputMax.valid) {
                        const unsigned int inputsIdx
                            = inputMax.ix + (inputMax.iy
                                + inputMax.channel * channelsHeight)
                                    * channelsWidth;

                        poolValue = inputs[inputsIdx + batchInputOffset];
                    }
                }
                else {
                    unsigned int ixMax = 0;
                    unsigned int iyMax = 0;
                    unsigned int channelMax = 0;
                    bool valid = false;

                    for (unsigned int channel = 0; channel < nbChannels;
                         ++channel)
                    {
                        if (maps != NULL && !maps[output + channel * nbOutputs])
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx)
                            {
                                const unsigned int inputsIdx
                                    = (ix + sx)
                                        + ((iy + sy) + channel * channelsHeight)
                                            * channelsWidth;

                                const float value = inputs[inputsIdx
                                    + batchInputOffset];

                                if (!valid || value > poolValue) {
                                    poolValue = value;
                                    valid = true;

                                    ixMax = ix + sx;
                                    iyMax = iy + sy;
                                    channelMax = channel;
                                }
                            }
                        }
                    }

                    argMax[outputsIdx].ix = ixMax;
                    argMax[outputsIdx].iy = iyMax;
                    argMax[outputsIdx].channel = channelMax;
                    argMax[outputsIdx].valid = valid;
                }

                if (beta != 0.0f) {
                    outputs[outputsIdx]
                        = alpha * poolValue
                          + beta * outputs[outputsIdx];
                }
                else {
                    outputs[outputsIdx] = alpha * poolValue;
                }
            }
        }
    }
}

template <>
__global__
void cudaPoolForwardMax_kernel<__half>(const __half alpha,
                                __half* inputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                const __half beta,
                                __half* outputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                bool useArgMax,
                                char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)max(
                    desc->padding[0] - (int)(ox * desc->stride[0]), 0);
                const unsigned int syMin = (unsigned int)max(
                    desc->padding[1] - (int)(oy * desc->stride[1]), 0);
                const unsigned int sxMax = min(
                    max(channelsWidth + desc->padding[0]
                                  - ox * desc->stride[0], 0),
                    desc->pool[0]);
                const unsigned int syMax = min(
                    max(channelsHeight + desc->padding[1]
                                  - oy * desc->stride[1], 0),
                    desc->pool[1]);

                const int ix = (int)(ox * desc->stride[0]) - desc->padding[0];
                const int iy = (int)(oy * desc->stride[1]) - desc->padding[1];

#if __CUDA_ARCH__ >= 530
                __half poolValue = __float2half(0.0f);
#else
                float poolValue = 0.0f;
#endif
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;

                // For each output, compute the pool value
                if (useArgMax) {
                    const N2D2::PoolCell_Frame_Kernels::ArgMax inputMax
                        = argMax[outputsIdx];

                    if (inputMax.valid) {
                        const unsigned int inputsIdx
                            = inputMax.ix + (inputMax.iy
                                + inputMax.channel * channelsHeight)
                                    * channelsWidth;
#if __CUDA_ARCH__ >= 530
                        poolValue = inputs[inputsIdx + batchInputOffset];
#else
                        poolValue = __half2float(inputs[inputsIdx
                                                         + batchInputOffset]);
#endif
                    }
                }
                else {
                    unsigned int ixMax = 0;
                    unsigned int iyMax = 0;
                    unsigned int channelMax = 0;
                    bool valid = false;

                    for (unsigned int channel = 0; channel < nbChannels;
                         ++channel)
                    {
                        if (maps != NULL && !maps[output + channel * nbOutputs])
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx)
                            {
                                const unsigned int inputsIdx
                                    = (ix + sx)
                                        + ((iy + sy) + channel * channelsHeight)
                                            * channelsWidth;
#if __CUDA_ARCH__ >= 530
                                const __half value
                                    = inputs[inputsIdx + batchInputOffset];

                                if (!valid || __hgt(value, poolValue)) {
                                    poolValue = value;
                                    valid = true;

                                    ixMax = ix + sx;
                                    iyMax = iy + sy;
                                    channelMax = channel;
                                }
#else
                                const float value
                                    = __half2float(inputs[inputsIdx
                                                           + batchInputOffset]);

                                if (!valid || value > poolValue) {
                                    poolValue = value;
                                    valid = true;

                                    ixMax = ix + sx;
                                    iyMax = iy + sy;
                                    channelMax = channel;
                                }
#endif
                            }
                        }
                    }

                    argMax[outputsIdx].ix = ixMax;
                    argMax[outputsIdx].iy = iyMax;
                    argMax[outputsIdx].channel = channelMax;
                    argMax[outputsIdx].valid = valid;
                }

#if __CUDA_ARCH__ >= 530
                if (!__heq(beta, __float2half(0.0f) )) {
                    outputs[outputsIdx] = __hadd(__hmul(alpha, poolValue),
                                            __hmul(beta, outputs[outputsIdx]));
                }
                else
                    outputs[outputsIdx] = __hmul(alpha, poolValue);
#else
                if (__half2float(beta) > 0.0f) {
                    outputs[outputsIdx] = __float2half( __half2float(alpha)
                                                       * poolValue
                      + __half2float(beta) * __half2float(outputs[outputsIdx]));
                }
                else {
                    outputs[outputsIdx] = __float2half(__half2float(alpha)
                                                       * poolValue);
                }
#endif
            }
        }
    }
}

// Backward
template <class T>
__global__
void cudaPoolBackwardAverage_kernel(const T alpha,
                                     T* diffInputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     unsigned int batchSize,
                                     const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                     const T beta,
                                     T* diffOutputs,
                                     unsigned int nbChannels,
                                     unsigned int channelsHeight,
                                     unsigned int channelsWidth,
                                     bool countIncludePadding,
                                     char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const unsigned int oxStride = desc->stride[0] * outputsWidth;
    const unsigned int oyStride = desc->stride[1] * outputsHeight;

    unsigned int* poolChannelsCount = new unsigned int[nbOutputs]();

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int channel = 0; channel < nbChannels; ++channel)
            poolChannelsCount[output] += (maps == NULL || maps[output
                                            + channel * nbOutputs]);
    }

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x)
    {
        for (unsigned int iy = threadIdx.y; iy < channelsHeight;
            iy += blockDim.y)
        {
            for (unsigned int ix = threadIdx.x; ix < channelsWidth;
                ix += blockDim.x)
            {
                const unsigned int ixPad = ix + desc->padding[0];
                const unsigned int iyPad = iy + desc->padding[1];
                const unsigned int sxMax = min(desc->pool[0], ixPad + 1);
                const unsigned int syMax = min(desc->pool[1], iyPad + 1);

                T poolGradient = 0.0f;

                for (unsigned int sy = iyPad % desc->stride[1],
                                  sx0 = ixPad % desc->stride[0];
                     sy < syMax;
                     sy += desc->stride[1])
                {
                    if (iyPad >= oyStride + sy)
                        continue;

                    for (unsigned int sx = sx0; sx < sxMax;
                         sx += desc->stride[0])
                    {
                        // Border conditions
                        if (ixPad >= oxStride + sx)
                            continue;

                        // Output node coordinates
                        const unsigned int ox = (ixPad - sx) / desc->stride[0];
                        const unsigned int oy = (iyPad - sy) / desc->stride[1];

                        for (unsigned int output = 0; output < nbOutputs;
                            ++output)
                        {
                            if (maps != NULL && !maps[output
                                + channel * nbOutputs])
                                continue;

                            const unsigned int outputsIdx
                                = ox + (oy + output * outputsHeight)
                                    * outputsWidth;
                            poolGradient
                                += diffInputs[outputsIdx + batchOutputOffset]
                                    / poolChannelsCount[output];
                        }
                    }
                }

                const unsigned int poolCount
                    = desc->pool[0] * desc->pool[1];

                const unsigned int inputsIdx
                    = ix + (iy + channel * channelsHeight) * channelsWidth
                        + batchInputOffset;

                if (beta != 0.0f) {
                    diffOutputs[inputsIdx]
                        = alpha * (poolGradient / poolCount)
                          + beta * diffOutputs[inputsIdx];
                }
                else {
                    diffOutputs[inputsIdx] = alpha * (poolGradient / poolCount);
                }
            }
        }
    }

    delete poolChannelsCount;
}

template <>
__global__
void cudaPoolBackwardAverage_kernel<__half>(const __half alpha,
                                     __half* diffInputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     unsigned int batchSize,
                                     const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                     const __half beta,
                                     __half* diffOutputs,
                                     unsigned int nbChannels,
                                     unsigned int channelsHeight,
                                     unsigned int channelsWidth,
                                     bool countIncludePadding,
                                     char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const unsigned int oxStride = desc->stride[0] * outputsWidth;
    const unsigned int oyStride = desc->stride[1] * outputsHeight;

    unsigned int* poolChannelsCount = new unsigned int[nbOutputs]();

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int channel = 0; channel < nbChannels; ++channel)
            poolChannelsCount[output] += (maps == NULL || maps[output
                                            + channel * nbOutputs]);
    }

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x)
    {
        for (unsigned int iy = threadIdx.y; iy < channelsHeight;
            iy += blockDim.y)
        {
            for (unsigned int ix = threadIdx.x; ix < channelsWidth;
                ix += blockDim.x)
            {
                const unsigned int ixPad = ix + desc->padding[0];
                const unsigned int iyPad = iy + desc->padding[1];
                const unsigned int sxMax = min(desc->pool[0], ixPad + 1);
                const unsigned int syMax = min(desc->pool[1], iyPad + 1);

#if __CUDA_ARCH__ >= 530
                __half poolGradient = __float2half(0.0f);
#else
                float poolGradient = 0.0f;
#endif

                for (unsigned int sy = iyPad % desc->stride[1],
                                  sx0 = ixPad % desc->stride[0];
                     sy < syMax;
                     sy += desc->stride[1])
                {
                    if (iyPad >= oyStride + sy)
                        continue;

                    for (unsigned int sx = sx0; sx < sxMax;
                         sx += desc->stride[0])
                    {
                        // Border conditions
                        if (ixPad >= oxStride + sx)
                            continue;

                        // Output node coordinates
                        const unsigned int ox = (ixPad - sx) / desc->stride[0];
                        const unsigned int oy = (iyPad - sy) / desc->stride[1];

                        for (unsigned int output = 0; output < nbOutputs;
                            ++output)
                        {
                            if (maps != NULL && !maps[output
                                + channel * nbOutputs])
                                continue;

                            const unsigned int outputsIdx
                                = ox + (oy + output * outputsHeight)
                                    * outputsWidth;
#if __CUDA_ARCH__ >= 530
                            poolGradient
                                = __hadd(__hmul(diffInputs[outputsIdx
                                                        + batchOutputOffset],
                                                __float2half(1.0f /
                                                    poolChannelsCount[output])),
                                         poolGradient);
#else
                            poolGradient += __half2float(diffInputs[outputsIdx
                                                           + batchOutputOffset])
                                    / poolChannelsCount[output];
#endif
                        }
                    }
                }

                const unsigned int poolCount
                    = desc->pool[0] * desc->pool[1];

                const unsigned int inputsIdx
                    = ix + (iy + channel * channelsHeight) * channelsWidth
                        + batchInputOffset;

#if __CUDA_ARCH__ >= 530
                if (! __heq(beta, __float2half(0.0f))) {
                    diffOutputs[inputsIdx] = __hadd(
                        __hmul(alpha, __hmul(poolGradient,
                                             __float2half(1.0f / poolCount))),
                        __hmul(beta, diffOutputs[inputsIdx]));
                }
                else {
                    diffOutputs[inputsIdx]
                        = __hmul(alpha, __hmul(poolGradient,
                                            __float2half(1.0f / poolCount)));
                }
#else
                if (__half2float(beta) != 0.0f) {
                    diffOutputs[inputsIdx] = __float2half(
                        __half2float(alpha) * (poolGradient / poolCount)
                        + __half2float(beta)
                                        * __half2float(diffOutputs[inputsIdx]));
                }
                else {
                    diffOutputs[inputsIdx] = __float2half( __half2float(alpha)
                                                * (poolGradient / poolCount));
                }
#endif
            }
        }
    }

    delete poolChannelsCount;
}

template <class T>
__global__
void cudaPoolBackwardMax_kernel(const T alpha,
                                 T* diffInputs,
                                 unsigned int nbOutputs,
                                 unsigned int outputsHeight,
                                 unsigned int outputsWidth,
                                 unsigned int batchSize,
                                 const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                 const T beta,
                                 T* diffOutputs,
                                 unsigned int nbChannels,
                                 unsigned int channelsHeight,
                                 unsigned int channelsWidth,
                                 N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                 char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const unsigned int oxStride = desc->stride[0] * outputsWidth;
    const unsigned int oyStride = desc->stride[1] * outputsHeight;

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x)
    {
        for (unsigned int iy = threadIdx.y; iy < channelsHeight;
            iy += blockDim.y)
        {
            for (unsigned int ix = threadIdx.x; ix < channelsWidth;
                ix += blockDim.x)
            {
                const unsigned int ixPad = ix + desc->padding[0];
                const unsigned int iyPad = iy + desc->padding[1];
                const unsigned int sxMax = min(desc->pool[0], ixPad + 1);
                const unsigned int syMax = min(desc->pool[1], iyPad + 1);

                T poolGradient = 0.0f;

                for (unsigned int sy = iyPad % desc->stride[1],
                                  sx0 = ixPad % desc->stride[0];
                     sy < syMax;
                     sy += desc->stride[1])
                {
                    if (iyPad >= oyStride + sy)
                        continue;

                    for (unsigned int sx = sx0; sx < sxMax;
                         sx += desc->stride[0])
                    {
                        // Border conditions
                        if (ixPad >= oxStride + sx)
                            continue;

                        // Output node coordinates
                        const unsigned int ox = (ixPad - sx) / desc->stride[0];
                        const unsigned int oy = (iyPad - sy) / desc->stride[1];

                        for (unsigned int output = 0; output < nbOutputs;
                            ++output)
                        {
                            if (maps != NULL && !maps[output
                                + channel * nbOutputs])
                                continue;

                            const unsigned int outputsIdx
                                = ox + (oy + output * outputsHeight)
                                    * outputsWidth + batchOutputOffset;
                            const N2D2::PoolCell_Frame_Kernels::ArgMax inputMax
                                = argMax[outputsIdx];

                            if (ix == inputMax.ix
                                && iy == inputMax.iy
                                && channel == inputMax.channel
                                && inputMax.valid)
                            {
                                poolGradient += diffInputs[outputsIdx];
                            }
                        }
                    }
                }

                const unsigned int inputsIdx
                    = ix + (iy + channel * channelsHeight) * channelsWidth
                        + batchInputOffset;

                if (beta != 0.0f) {
                    diffOutputs[inputsIdx]
                        = alpha * poolGradient
                          + beta * diffOutputs[inputsIdx];
                }
                else {
                    diffOutputs[inputsIdx] = alpha * poolGradient;
                }
            }
        }
    }
}

template <>
__global__
void cudaPoolBackwardMax_kernel<__half>(const __half alpha,
                                 __half* diffInputs,
                                 unsigned int nbOutputs,
                                 unsigned int outputsHeight,
                                 unsigned int outputsWidth,
                                 unsigned int batchSize,
                                 const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                 const __half beta,
                                 __half* diffOutputs,
                                 unsigned int nbChannels,
                                 unsigned int channelsHeight,
                                 unsigned int channelsWidth,
                                 N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                 char* maps)
{
    const unsigned int batchInputOffset = blockIdx.z * nbChannels
                                          * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const unsigned int oxStride = desc->stride[0] * outputsWidth;
    const unsigned int oyStride = desc->stride[1] * outputsHeight;

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x)
    {
        for (unsigned int iy = threadIdx.y; iy < channelsHeight;
            iy += blockDim.y)
        {
            for (unsigned int ix = threadIdx.x; ix < channelsWidth;
                ix += blockDim.x)
            {
                const unsigned int ixPad = ix + desc->padding[0];
                const unsigned int iyPad = iy + desc->padding[1];
                const unsigned int sxMax = min(desc->pool[0], ixPad + 1);
                const unsigned int syMax = min(desc->pool[1], iyPad + 1);

#if __CUDA_ARCH__ >= 530
                __half poolGradient = __float2half(0.0f);
#else
                float poolGradient = 0.0f;
#endif

                for (unsigned int sy = iyPad % desc->stride[1],
                                  sx0 = ixPad % desc->stride[0];
                     sy < syMax;
                     sy += desc->stride[1])
                {
                    if (iyPad >= oyStride + sy)
                        continue;

                    for (unsigned int sx = sx0; sx < sxMax;
                         sx += desc->stride[0])
                    {
                        // Border conditions
                        if (ixPad >= oxStride + sx)
                            continue;

                        // Output node coordinates
                        const unsigned int ox = (ixPad - sx) / desc->stride[0];
                        const unsigned int oy = (iyPad - sy) / desc->stride[1];

                        for (unsigned int output = 0; output < nbOutputs;
                            ++output)
                        {
                            if (maps != NULL && !maps[output
                                + channel * nbOutputs])
                                continue;

                            const unsigned int outputsIdx
                                = ox + (oy + output * outputsHeight)
                                    * outputsWidth + batchOutputOffset;
                            const N2D2::PoolCell_Frame_Kernels::ArgMax inputMax
                                = argMax[outputsIdx];

                            if (ix == inputMax.ix
                                && iy == inputMax.iy
                                && channel == inputMax.channel
                                && inputMax.valid)
                            {
#if __CUDA_ARCH__ >= 530
                                poolGradient = __hadd(poolGradient,
                                                      diffInputs[outputsIdx]);
#else
                                poolGradient
                                    += __half2float(diffInputs[outputsIdx]);
#endif
                            }
                        }
                    }
                }

                const unsigned int inputsIdx
                    = ix + (iy + channel * channelsHeight) * channelsWidth
                        + batchInputOffset;

#if __CUDA_ARCH__ >= 530
                if (!__heq(beta, __float2half(0.0f)) ) {
                    diffOutputs[inputsIdx]
                        = __hadd(__hmul(alpha, poolGradient),
                                 __hmul(beta, diffOutputs[inputsIdx]));
                }
                else {
                    diffOutputs[inputsIdx] = __hmul(alpha, poolGradient);
                }
#else
                if (__half2float(beta) != 0.0f) {
                    diffOutputs[inputsIdx] = __float2half(
                        __half2float(alpha) * poolGradient
                        + __half2float(beta)
                                    * __half2float(diffOutputs[inputsIdx]));
                }
                else {
                    diffOutputs[inputsIdx] = __float2half(__half2float(alpha)
                                                          * poolGradient);
                }
#endif
            }
        }
    }
}

namespace N2D2 {

template <class T>
void cudaPoolForwardAverage(const cudaDeviceProp& deviceProp,
                                   T alpha,
                                   T* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const N2D2::PoolCell_Frame_Kernels
                                    ::Descriptor* desc,
                                   T beta,
                                   T* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth,
                                   bool countIncludePadding,
                                   char* maps)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;

    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);
    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaPoolForwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type&>(alpha),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(inputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           reinterpret_cast<typename Cuda::cuda_type<T>::type&>(beta),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaPoolForwardMax(const cudaDeviceProp& deviceProp,
                               T alpha,
                               T* inputs,
                               unsigned int nbChannels,
                               unsigned int channelsHeight,
                               unsigned int channelsWidth,
                               unsigned int batchSize,
                               const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                desc,
                               T beta,
                               T* outputs,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                               bool useArgMax,
                               char* maps)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaPoolForwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type&>(alpha),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(inputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           reinterpret_cast<typename Cuda::cuda_type<T>::type&>(beta),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           argMax,
           useArgMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaPoolBackwardAverage(const cudaDeviceProp& deviceProp,
                                    T alpha,
                                    T* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    T beta,
                                    T* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    bool countIncludePadding,
                                    char* maps)
{
    if (!countIncludePadding) {
        throw std::runtime_error("PoolCell_Frame_CUDA_Kernels::"
            "backwardAverage() exclude padding not implemented");
    }

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaPoolBackwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type&>(alpha),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffInputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           reinterpret_cast<typename Cuda::cuda_type<T>::type&>(beta),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffOutputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaPoolBackwardMax(const cudaDeviceProp& deviceProp,
                                T alpha,
                                T* diffInputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                T beta,
                                T* diffOutputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                char* maps)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaPoolBackwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<typename Cuda::cuda_type<T>::type&>(alpha),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffInputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           reinterpret_cast<typename Cuda::cuda_type<T>::type&>(beta),
           reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffOutputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           argMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaPoolForwardAverage(const cudaDeviceProp& deviceProp,
                                   half_float::half alpha,
                                   half_float::half* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const N2D2::PoolCell_Frame_Kernels
                                    ::Descriptor* desc,
                                   half_float::half beta,
                                   half_float::half* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth,
                                   bool countIncludePadding,
                                   char* maps);
template void cudaPoolForwardAverage(const cudaDeviceProp& deviceProp,
                                   float alpha,
                                   float* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const N2D2::PoolCell_Frame_Kernels
                                    ::Descriptor* desc,
                                   float beta,
                                   float* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth,
                                   bool countIncludePadding,
                                   char* maps);
template void cudaPoolForwardAverage(const cudaDeviceProp& deviceProp,
                                   double alpha,
                                   double* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const N2D2::PoolCell_Frame_Kernels
                                    ::Descriptor* desc,
                                   double beta,
                                   double* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth,
                                   bool countIncludePadding,
                                   char* maps);

template void cudaPoolForwardMax(const cudaDeviceProp& deviceProp,
                               half_float::half alpha,
                               half_float::half* inputs,
                               unsigned int nbChannels,
                               unsigned int channelsHeight,
                               unsigned int channelsWidth,
                               unsigned int batchSize,
                               const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                desc,
                               half_float::half beta,
                               half_float::half* outputs,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                               bool useArgMax,
                               char* maps);
template void cudaPoolForwardMax(const cudaDeviceProp& deviceProp,
                               float alpha,
                               float* inputs,
                               unsigned int nbChannels,
                               unsigned int channelsHeight,
                               unsigned int channelsWidth,
                               unsigned int batchSize,
                               const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                desc,
                               float beta,
                               float* outputs,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                               bool useArgMax,
                               char* maps);
template void cudaPoolForwardMax(const cudaDeviceProp& deviceProp,
                               double alpha,
                               double* inputs,
                               unsigned int nbChannels,
                               unsigned int channelsHeight,
                               unsigned int channelsWidth,
                               unsigned int batchSize,
                               const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                desc,
                               double beta,
                               double* outputs,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                               bool useArgMax,
                               char* maps);

template void cudaPoolBackwardAverage(const cudaDeviceProp& deviceProp,
                                    half_float::half alpha,
                                    half_float::half* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    half_float::half beta,
                                    half_float::half* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    bool countIncludePadding,
                                    char* maps);
template void cudaPoolBackwardAverage(const cudaDeviceProp& deviceProp,
                                    float alpha,
                                    float* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    float beta,
                                    float* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    bool countIncludePadding,
                                    char* maps);
template void cudaPoolBackwardAverage(const cudaDeviceProp& deviceProp,
                                    double alpha,
                                    double* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    double beta,
                                    double* diffOutputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    bool countIncludePadding,
                                    char* maps);

template void cudaPoolBackwardMax(const cudaDeviceProp& deviceProp,
                                half_float::half alpha,
                                half_float::half* diffInputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                half_float::half beta,
                                half_float::half* diffOutputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                char* maps);
template void cudaPoolBackwardMax(const cudaDeviceProp& deviceProp,
                                float alpha,
                                float* diffInputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                float beta,
                                float* diffOutputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                char* maps);
template void cudaPoolBackwardMax(const cudaDeviceProp& deviceProp,
                                double alpha,
                                double* diffInputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                double beta,
                                double* diffOutputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                char* maps);

}
