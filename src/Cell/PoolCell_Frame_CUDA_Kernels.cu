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

// Forward
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
                    desc->paddingX - (int)(ox * desc->strideX), 0);
                const unsigned int syMin = (unsigned int)max(
                    desc->paddingY - (int)(oy * desc->strideY), 0);
                const unsigned int sxMax = min(
                    max(channelsWidth + desc->paddingX
                                  - ox * desc->strideX, 0),
                    desc->poolWidth);
                const unsigned int syMax = min(
                    max(channelsHeight + desc->paddingY
                                  - oy * desc->strideY, 0),
                    desc->poolHeight);

                const int ix = (int)(ox * desc->strideX) - desc->paddingX;
                const int iy = (int)(oy * desc->strideY) - desc->paddingY;

                // For each output, compute the pool value
                float poolValue = 0.0;
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
                        ? (desc->poolWidth * desc->poolHeight)
                        : (sxMax - sxMin)*(syMax - syMin);
                }

                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth;
                outputs[outputsIdx + batchOutputOffset]
                    = alpha * ((poolCount > 0) ?
                                  (poolValue / poolCount) : 0.0)
                      + beta * outputs[outputsIdx + batchOutputOffset];
            }
        }
    }
}

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
                    desc->paddingX - (int)(ox * desc->strideX), 0);
                const unsigned int syMin = (unsigned int)max(
                    desc->paddingY - (int)(oy * desc->strideY), 0);
                const unsigned int sxMax = min(
                    max(channelsWidth + desc->paddingX
                                  - ox * desc->strideX, 0),
                    desc->poolWidth);
                const unsigned int syMax = min(
                    max(channelsHeight + desc->paddingY
                                  - oy * desc->strideY, 0),
                    desc->poolHeight);

                const int ix = (int)(ox * desc->strideX) - desc->paddingX;
                const int iy = (int)(oy * desc->strideY) - desc->paddingY;

                float poolValue = 0.0;

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

                outputs[outputsIdx]
                    = alpha * poolValue
                      + beta * outputs[outputsIdx];
            }
        }
    }
}

// Backward
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

    const unsigned int oxStride = desc->strideX * outputsWidth;
    const unsigned int oyStride = desc->strideY * outputsHeight;

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
                const unsigned int ixPad = ix + desc->paddingX;
                const unsigned int iyPad = iy + desc->paddingY;
                const unsigned int sxMax = min(desc->poolWidth, ixPad + 1);
                const unsigned int syMax = min(desc->poolHeight, iyPad + 1);

                float poolGradient = 0.0;

                for (unsigned int sy = iyPad % desc->strideY,
                                  sx0 = ixPad % desc->strideX;
                     sy < syMax;
                     sy += desc->strideY)
                {
                    if (iyPad >= oyStride + sy)
                        continue;

                    for (unsigned int sx = sx0; sx < sxMax;
                         sx += desc->strideX)
                    {
                        // Border conditions
                        if (ixPad >= oxStride + sx)
                            continue;

                        // Output node coordinates
                        const unsigned int ox = (ixPad - sx) / desc->strideX;
                        const unsigned int oy = (iyPad - sy) / desc->strideY;

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
                    = desc->poolWidth * desc->poolHeight;

                const unsigned int inputsIdx
                    = ix + (iy + channel * channelsHeight) * channelsWidth
                        + batchInputOffset;
                diffOutputs[inputsIdx]
                    = alpha * (poolGradient / poolCount)
                      + beta * diffOutputs[inputsIdx];
            }
        }
    }

    delete poolChannelsCount;
}

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

    const unsigned int oxStride = desc->strideX * outputsWidth;
    const unsigned int oyStride = desc->strideY * outputsHeight;

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x)
    {
        for (unsigned int iy = threadIdx.y; iy < channelsHeight;
            iy += blockDim.y)
        {
            for (unsigned int ix = threadIdx.x; ix < channelsWidth;
                ix += blockDim.x)
            {
                const unsigned int ixPad = ix + desc->paddingX;
                const unsigned int iyPad = iy + desc->paddingY;
                const unsigned int sxMax = min(desc->poolWidth, ixPad + 1);
                const unsigned int syMax = min(desc->poolHeight, iyPad + 1);

                float poolGradient = 0.0;

                for (unsigned int sy = iyPad % desc->strideY,
                                  sx0 = ixPad % desc->strideX;
                     sy < syMax;
                     sy += desc->strideY)
                {
                    if (iyPad >= oyStride + sy)
                        continue;

                    for (unsigned int sx = sx0; sx < sxMax;
                         sx += desc->strideX)
                    {
                        // Border conditions
                        if (ixPad >= oxStride + sx)
                            continue;

                        // Output node coordinates
                        const unsigned int ox = (ixPad - sx) / desc->strideX;
                        const unsigned int oy = (iyPad - sy) / desc->strideY;

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
                diffOutputs[inputsIdx]
                    = alpha * poolGradient
                      + beta * diffOutputs[inputsIdx];
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
