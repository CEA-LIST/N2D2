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
//Half
__global__
void cudaHPoolForwardAverage_kernel(const __half alpha,
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
                __half poolValue(0.0f);
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
                __half finalOutput(0.0f);

                if (poolCount > 0) {
                    finalOutput = __hdiv( __hmul(poolValue, alpha),
                                         __float2half(float(poolCount)));
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
//Float
__global__
void cudaSPoolForwardAverage_kernel(const float alpha,
                                    float* inputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    const float beta,
                                    float* outputs,
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
                float poolValue = 0.0f;
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
//Double
__global__
void cudaDPoolForwardAverage_kernel(const double alpha,
                                    double* inputs,
                                    unsigned int nbChannels,
                                    unsigned int channelsHeight,
                                    unsigned int channelsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    const double beta,
                                    double* outputs,
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
                double poolValue = 0.0;
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

                if (beta != 0.0) {
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

////Forward MAX
//Half
__global__
void cudaHPoolForwardMax_kernel(const __half alpha,
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
                __half poolValue(0.0f);
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
                        poolValue = __half2float(inputs[inputsId
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
//Float
__global__
void cudaSPoolForwardMax_kernel(const float alpha,
                                float* inputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                const float beta,
                                float* outputs,
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

                float poolValue = 0.0f;

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
//Double
__global__
void cudaDPoolForwardMax_kernel(const double alpha,
                                double* inputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                const double beta,
                                double* outputs,
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

                double poolValue = 0.0;

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

                                const double value = inputs[inputsIdx
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

                if (beta != 0.0) {
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

// Backward
//Half
__global__
void cudaHPoolBackwardAverage_kernel(const __half alpha,
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
                __half poolGradient(0.0f);
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
                                = __hadd(__hdiv(diffInputs[outputsIdx
                                                        + batchOutputOffset],
                                                __float2half((float)
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
                        __hmul(alpha, __hdiv(poolGradient,
                                             __float2half((float) poolCount))),
                        __hmul(beta, diffOutputs[inputsIdx]));
                }
                else {
                    diffOutputs[inputsIdx]
                        = __hmul(alpha, __hdiv(poolGradient,
                                            __float2half((float) poolCount)));
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
//Float
__global__
void cudaSPoolBackwardAverage_kernel(const float alpha,
                                     float* diffInputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     unsigned int batchSize,
                                     const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                     const float beta,
                                     float* diffOutputs,
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

                float poolGradient = 0.0f;

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
//Double
__global__
void cudaDPoolBackwardAverage_kernel(const double alpha,
                                     double* diffInputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     unsigned int batchSize,
                                     const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                     const double beta,
                                     double* diffOutputs,
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

                double poolGradient = 0.0;

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

                if (beta != 0.0) {
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


//Half
__global__
void cudaHPoolBackwardMax_kernel(const __half alpha,
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

                __half poolGradient(0.0f);
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
//Float
__global__
void cudaSPoolBackwardMax_kernel(const float alpha,
                                 float* diffInputs,
                                 unsigned int nbOutputs,
                                 unsigned int outputsHeight,
                                 unsigned int outputsWidth,
                                 unsigned int batchSize,
                                 const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                 const float beta,
                                 float* diffOutputs,
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

                float poolGradient = 0.0f;

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
//Double
__global__
void cudaDPoolBackwardMax_kernel(const double alpha,
                                 double* diffInputs,
                                 unsigned int nbOutputs,
                                 unsigned int outputsHeight,
                                 unsigned int outputsWidth,
                                 unsigned int batchSize,
                                 const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                 const double beta,
                                 double* diffOutputs,
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

                double poolGradient = 0.0;

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

                if (beta != 0.0) {
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


static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}
//Half
void N2D2::cudaHPoolForwardAverage(half_float::half alpha,
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
                                   char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaHPoolForwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<__half&>(alpha),
           reinterpret_cast<__half*>(inputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           reinterpret_cast<__half&>(beta),
           reinterpret_cast<__half*>(outputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Float
void N2D2::cudaSPoolForwardAverage(const float alpha,
                                   float* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const N2D2::PoolCell_Frame_Kernels
                                    ::Descriptor* desc,
                                   const float beta,
                                   float* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth,
                                   bool countIncludePadding,
                                   char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSPoolForwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Double
void N2D2::cudaDPoolForwardAverage(const double alpha,
                                   double* inputs,
                                   unsigned int nbChannels,
                                   unsigned int channelsHeight,
                                   unsigned int channelsWidth,
                                   unsigned int batchSize,
                                   const N2D2::PoolCell_Frame_Kernels
                                    ::Descriptor* desc,
                                   const double beta,
                                   double* outputs,
                                   unsigned int nbOutputs,
                                   unsigned int outputsHeight,
                                   unsigned int outputsWidth,
                                   bool countIncludePadding,
                                   char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaDPoolForwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


//Half
void N2D2::cudaHPoolForwardMax(half_float::half alpha,
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
                               char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaHPoolForwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<__half&>(alpha),
           reinterpret_cast<__half*>(inputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           reinterpret_cast<__half&>(beta),
           reinterpret_cast<__half*>(outputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           argMax,
           useArgMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Float
void N2D2::cudaSPoolForwardMax(const float alpha,
                               float* inputs,
                               unsigned int nbChannels,
                               unsigned int channelsHeight,
                               unsigned int channelsWidth,
                               unsigned int batchSize,
                               const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                desc,
                               const float beta,
                               float* outputs,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                               bool useArgMax,
                               char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSPoolForwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           argMax,
           useArgMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Double
void N2D2::cudaDPoolForwardMax(const double alpha,
                               double* inputs,
                               unsigned int nbChannels,
                               unsigned int channelsHeight,
                               unsigned int channelsWidth,
                               unsigned int batchSize,
                               const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                desc,
                               const double beta,
                               double* outputs,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                               bool useArgMax,
                               char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, outputsWidth));

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaDPoolForwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           desc,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           argMax,
           useArgMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

//Half
void N2D2::cudaHPoolBackwardAverage(half_float::half alpha,
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
                                    char* maps)
{
    if (!countIncludePadding) {
        throw std::runtime_error("PoolCell_Frame_CUDA_Kernels::"
            "backwardAverage() exclude padding not implemented");
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaHPoolBackwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<__half&>(alpha),
           reinterpret_cast<__half*>(diffInputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           reinterpret_cast<__half&>(beta),
           reinterpret_cast<__half*>(diffOutputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Float
void N2D2::cudaSPoolBackwardAverage(const float alpha,
                                    float* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    const float beta,
                                    float* diffOutputs,
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

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSPoolBackwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Double
void N2D2::cudaDPoolBackwardAverage(const double alpha,
                                    double* diffInputs,
                                    unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    unsigned int batchSize,
                                    const N2D2::PoolCell_Frame_Kernels
                                        ::Descriptor* desc,
                                    const double beta,
                                    double* diffOutputs,
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

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaDPoolBackwardAverage_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           countIncludePadding,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

//Half
void N2D2::cudaHPoolBackwardMax(half_float::half alpha,
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
                                char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaHPoolBackwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<__half&>(alpha),
           reinterpret_cast<__half*>(diffInputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           reinterpret_cast<__half&>(beta),
           reinterpret_cast<__half*>(diffOutputs),
           nbChannels,
           channelsHeight,
           channelsWidth,
           argMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Float
void N2D2::cudaSPoolBackwardMax(const float alpha,
                                float* diffInputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                const float beta,
                                float* diffOutputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSPoolBackwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           argMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
//Double
void N2D2::cudaDPoolBackwardMax(const double alpha,
                                double* diffInputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int batchSize,
                                const N2D2::PoolCell_Frame_Kernels::Descriptor*
                                    desc,
                                const double beta,
                                double* diffOutputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                N2D2::PoolCell_Frame_Kernels::ArgMax* argMax,
                                char* maps)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (channelsWidth * channelsHeight < maxSize)
                                       ? channelsWidth * channelsHeight
                                       : maxSize;
    const unsigned int groupWidth
        = min(prefMultiple, nextDivisor(groupSize, channelsWidth));

    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaDPoolBackwardMax_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (alpha,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           desc,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           argMax,
           maps);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
