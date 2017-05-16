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

#include "Cell/ROIPoolingCell_Frame_CUDA_Kernels.hpp"

// Forward
__global__
void cudaSROIPoolingForwardAverage_kernel(const float alpha,
                                          float* proposals,
                                          unsigned int nbProposals,
                                          unsigned int inputSizeY,
                                          unsigned int inputSizeX,
                                          float* inputs,
                                          unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth,
                                          unsigned int batchSize,
                                          const float beta,
                                          float* outputs,
                                          unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth)
{
    const unsigned int batchProposalsOffset = blockIdx.z * 4;
    const unsigned int batchInputOffset = (blockIdx.z / nbProposals)
                                * nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const float xRatio = inputSizeX / (float)channelsWidth;
    const float yRatio = inputSizeY / (float)channelsHeight;

    float x = proposals[0 + batchProposalsOffset] / xRatio;
    float y = proposals[1 + batchProposalsOffset] / yRatio;
    float w = proposals[2 + batchProposalsOffset] / xRatio;
    float h = proposals[3 + batchProposalsOffset] / yRatio;

    // Crop ROI to image boundaries
    if (x < 0) {
        w+= x;
        x = 0;
    }
    if (y < 0) {
        h+= y;
        y = 0;
    }
    if (x + w > (int)channelsWidth)
        w = channelsWidth - x;
    if (y + h > (int)channelsHeight)
        h = channelsHeight - y;

    const float poolWidth = w / outputsWidth;
    const float poolHeight = h / outputsHeight;

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)(x
                                        + ox * poolWidth);
                const unsigned int sxMax = (unsigned int)(x
                                        + (ox + 1) * poolWidth);
                const unsigned int syMin = (unsigned int)(y
                                        + oy * poolHeight);
                const unsigned int syMax = (unsigned int)(y
                                        + (oy + 1) * poolHeight);

                // For each output, compute the pool value
                float poolValue = 0.0;
                unsigned int poolCount = 0;

                for (unsigned int sy = syMin; sy < syMax; ++sy) {
                    for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                        const unsigned int inputsIdx
                            = sx
                                + (sy + output * channelsHeight)
                                    * channelsWidth;

                        poolValue += inputs[inputsIdx + batchInputOffset];
                    }
                }

                poolCount += (sxMax - sxMin)*(syMax - syMin);

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
void cudaSROIPoolingForwardMax_kernel(const float alpha,
                                      float* proposals,
                                      unsigned int nbProposals,
                                      unsigned int inputSizeY,
                                      unsigned int inputSizeX,
                                      float* inputs,
                                      unsigned int nbChannels,
                                      unsigned int channelsHeight,
                                      unsigned int channelsWidth,
                                      unsigned int batchSize,
                                      const float beta,
                                      float* outputs,
                                      unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
{
    const unsigned int batchProposalsOffset = blockIdx.z * 4;
    const unsigned int batchInputOffset = (blockIdx.z / nbProposals)
                                * nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const float xRatio = inputSizeX / (float)channelsWidth;
    const float yRatio = inputSizeY / (float)channelsHeight;

    float x = proposals[0 + batchProposalsOffset] / xRatio;
    float y = proposals[1 + batchProposalsOffset] / yRatio;
    float w = proposals[2 + batchProposalsOffset] / xRatio;
    float h = proposals[3 + batchProposalsOffset] / yRatio;

    // Crop ROI to image boundaries
    if (x < 0) {
        w+= x;
        x = 0;
    }
    if (y < 0) {
        h+= y;
        y = 0;
    }
    if (x + w > (int)channelsWidth)
        w = channelsWidth - x;
    if (y + h > (int)channelsHeight)
        h = channelsHeight - y;

    const float poolWidth = w / outputsWidth;
    const float poolHeight = h / outputsHeight;

    for (unsigned int output = blockIdx.x; output < nbOutputs;
         output += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                const unsigned int sxMin = (unsigned int)(x
                                        + ox * poolWidth);
                const unsigned int sxMax = (unsigned int)(x
                                        + (ox + 1) * poolWidth);
                const unsigned int syMin = (unsigned int)(y
                                        + oy * poolHeight);
                const unsigned int syMax = (unsigned int)(y
                                        + (oy + 1) * poolHeight);

                // For each output, compute the pool value
                float poolValue = 0.0;

                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;

                unsigned int ixMax = 0;
                unsigned int iyMax = 0;
                bool valid = false;

                for (unsigned int sy = syMin; sy < syMax; ++sy) {
                    for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                        const unsigned int inputsIdx
                            = sx
                                + (sy + output * channelsHeight)
                                    * channelsWidth;

                        const float value = inputs[inputsIdx
                                                + batchInputOffset];

                        if (!valid || value > poolValue) {
                            poolValue = value;
                            valid = true;

                            ixMax = sx;
                            iyMax = sy;
                        }
                    }
                }

                argMax[outputsIdx].ix = ixMax;
                argMax[outputsIdx].iy = iyMax;
                argMax[outputsIdx].channel = output;
                argMax[outputsIdx].valid = valid;

                outputs[outputsIdx]
                    = alpha * poolValue
                      + beta * outputs[outputsIdx];
            }
        }
    }
}

// Backward
__global__
void cudaSROIPoolingBackwardAverage_kernel(const float alpha,
                                          float* proposals,
                                          unsigned int nbProposals,
                                          unsigned int inputSizeY,
                                          unsigned int inputSizeX,
                                          float* diffInputs,
                                          unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int batchSize,
                                          const float beta,
                                          float* diffOutputs,
                                          unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth)
{
    //TODO
}

__global__
void cudaSROIPoolingBackwardMax_kernel(const float alpha,
                                      float* proposals,
                                      unsigned int nbProposals,
                                      unsigned int inputSizeY,
                                      unsigned int inputSizeX,
                                      float* diffInputs,
                                      unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      unsigned int batchSize,
                                      const float /*beta*/,
                                      float* diffOutputs,
                                      unsigned int nbChannels,
                                      unsigned int channelsHeight,
                                      unsigned int channelsWidth,
                                      N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
{
    const unsigned int batchProposalsOffset = blockIdx.z * 4;
    const unsigned int batchInputOffset = (blockIdx.z / nbProposals)
                                * nbChannels * channelsHeight * channelsWidth;
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    //const float betaBatch = (blockIdx.z % nbProposals == 0)
    //                            ? beta : 1.0;

    const float xRatio = inputSizeX / (float)channelsWidth;
    const float yRatio = inputSizeY / (float)channelsHeight;

    float x = proposals[0 + batchProposalsOffset] / xRatio;
    float y = proposals[1 + batchProposalsOffset] / yRatio;
    float w = proposals[2 + batchProposalsOffset] / xRatio;
    float h = proposals[3 + batchProposalsOffset] / yRatio;

    // Crop ROI to image boundaries
    if (x < 0) {
        w+= x;
        x = 0;
    }
    if (y < 0) {
        h+= y;
        y = 0;
    }
    if (x + w > (int)channelsWidth)
        w = channelsWidth - x;
    if (y + h > (int)channelsHeight)
        h = channelsHeight - y;

    const float poolWidth = w / outputsWidth;
    const float poolHeight = h / outputsHeight;

    const unsigned int ixMin = (unsigned int)(x);
    const unsigned int ixMax = (unsigned int)(x + w);
    const unsigned int iyMin = (unsigned int)(y);
    const unsigned int iyMax = (unsigned int)(y + h);

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x)
    {
        for (unsigned int iy = threadIdx.y; iy < channelsHeight;
            iy += blockDim.y)
        {
            for (unsigned int ix = threadIdx.x; ix < channelsWidth;
                ix += blockDim.x)
            {
                if (ix >= ixMin && ix < ixMax
                    && iy >= iyMin && iy < iyMax)
                {
                    const unsigned int ox
                        = (unsigned int)((ix - ixMin + 0.5) / poolWidth);
                    const unsigned int oy
                        = (unsigned int)((iy - iyMin + 0.5) / poolHeight);

                    const unsigned int outputsIdx
                        = ox + (oy + channel * outputsHeight)
                            * outputsWidth + batchOutputOffset;
                    const N2D2::PoolCell_Frame_Kernels::ArgMax inputMax
                        = argMax[outputsIdx];

                    if (ix == inputMax.ix
                        && iy == inputMax.iy
                        && inputMax.valid)
                    {
                        const unsigned int inputsIdx
                            = ix + (iy + channel * channelsHeight)
                                * channelsWidth + batchInputOffset;

                        atomicAdd(diffOutputs + inputsIdx,
                                  alpha * diffInputs[outputsIdx]);
                    }
                }
/*
                diffOutputs[inputsIdx]
                    = alpha * poolGradient
                      + betaBatch * diffOutputs[inputsIdx];
*/
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

void N2D2::cudaSROIPoolingForwardAverage(const float alpha,
                                         float* proposals,
                                         unsigned int nbProposals,
                                         unsigned int inputSizeY,
                                         unsigned int inputSizeX,
                                         float* inputs,
                                         unsigned int nbChannels,
                                         unsigned int channelsHeight,
                                         unsigned int channelsWidth,
                                         unsigned int batchSize,
                                         const float beta,
                                         float* outputs,
                                         unsigned int nbOutputs,
                                         unsigned int outputsHeight,
                                         unsigned int outputsWidth)
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

    cudaSROIPoolingForwardAverage_kernel << <blocksPerGrid, threadsPerBlocks>>
        > (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth);
}

void N2D2::cudaSROIPoolingForwardMax(const float alpha,
                                     float* proposals,
                                     unsigned int nbProposals,
                                     unsigned int inputSizeY,
                                     unsigned int inputSizeX,
                                     float* inputs,
                                     unsigned int nbChannels,
                                     unsigned int channelsHeight,
                                     unsigned int channelsWidth,
                                     unsigned int batchSize,
                                     const float beta,
                                     float* outputs,
                                     unsigned int nbOutputs,
                                     unsigned int outputsHeight,
                                     unsigned int outputsWidth,
                                     N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
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

    cudaSROIPoolingForwardMax_kernel << <blocksPerGrid, threadsPerBlocks>>
        > (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           inputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize,
           beta,
           outputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           argMax);
}

void N2D2::cudaSROIPoolingBackwardAverage(const float alpha,
                                          float* proposals,
                                          unsigned int nbProposals,
                                          unsigned int inputSizeY,
                                          unsigned int inputSizeX,
                                          float* diffInputs,
                                          unsigned int nbOutputs,
                                          unsigned int outputsHeight,
                                          unsigned int outputsWidth,
                                          unsigned int batchSize,
                                          const float beta,
                                          float* diffOutputs,
                                          unsigned int nbChannels,
                                          unsigned int channelsHeight,
                                          unsigned int channelsWidth)
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

    cudaSROIPoolingBackwardAverage_kernel << <blocksPerGrid, threadsPerBlocks>>
        > (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth);
}

void N2D2::cudaSROIPoolingBackwardMax(const float alpha,
                                      float* proposals,
                                      unsigned int nbProposals,
                                      unsigned int inputSizeY,
                                      unsigned int inputSizeX,
                                      float* diffInputs,
                                      unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      unsigned int batchSize,
                                      const float beta,
                                      float* diffOutputs,
                                      unsigned int nbChannels,
                                      unsigned int channelsHeight,
                                      unsigned int channelsWidth,
                                      N2D2::PoolCell_Frame_Kernels::ArgMax*
                                        argMax)
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

    cudaSROIPoolingBackwardMax_kernel << <blocksPerGrid, threadsPerBlocks>>
        > (alpha,
           proposals,
           nbProposals,
           inputSizeY,
           inputSizeX,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           beta,
           diffOutputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           argMax);
}
