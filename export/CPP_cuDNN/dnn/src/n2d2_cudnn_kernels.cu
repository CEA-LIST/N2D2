/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "n2d2_cudnn_kernels.hpp"

static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}

__global__ void cudaSBNPropagate_kernel(const DATA_T* inputs,
                                        float* bias,
                                        float* variance,
                                        float* mean,
                                        float* scale,
                                        float epsilon,
                                        DATA_T* outputs,
                                        unsigned int nbChannels,
                                        unsigned int channelsHeight,
                                        unsigned int channelsWidth,
                                        unsigned int batchSize)
{
    const unsigned int batchOutputOffset
        = blockIdx.z * nbChannels * channelsHeight
          * channelsWidth; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbChannels;
         output += gridDim.x) {
        const float var = sqrt(variance[output] + epsilon);

        for (unsigned int oy = threadIdx.y; oy < channelsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < channelsWidth;
                 ox += blockDim.x) {
                const unsigned int outputsIdx
                    = ox + (oy + output * channelsHeight) * channelsWidth;

                const float normalized
                    = ((float)(inputs[outputsIdx + batchOutputOffset])
                       - mean[output]) / var;
                const float sAs
                    = scale[output] * normalized
                      + bias[output]; // Scale and Shift normalized value

                outputs[batchOutputOffset + outputsIdx]
                    = max((float)0, sAs); // FOR RECTIFIER
            }
        }
    }
}

void cudaSBNPropagate(const DATA_T* inputs,
                      float* bias,
                      float* variance,
                      float* mean,
                      float* scale,
                      float epsilon,
                      DATA_T* outputs,
                      unsigned int nbChannels,
                      unsigned int channelsHeight,
                      unsigned int channelsWidth,
                      unsigned int batchSize)
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

    const uint3 blocksPerGrid = {nbChannels, 1, batchSize};
    const uint3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSBNPropagate_kernel << <blocksPerGrid, threadsPerBlocks>>
        > (inputs,
           bias,
           variance,
           mean,
           scale,
           epsilon,
           outputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           batchSize);
}

__global__ void cudaSFMPPropagate_kernel(const DATA_T* inputs,
                                         unsigned int* gridX,
                                         unsigned int* gridY,
                                         DATA_T* outputs,
                                         unsigned int nbChannels,
                                         unsigned int channelsHeight,
                                         unsigned int channelsWidth,
                                         unsigned int nbOutputs,
                                         unsigned int outputsHeight,
                                         unsigned int outputsWidth,
                                         unsigned int batchSize,
                                         bool overlapping)
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
                 ox += blockDim.x) {
                // For each output, compute the pool value
                float poolValue = -FLT_MAX;
                /*
                                unsigned int channelMax = 0;
                                unsigned int ixMax = 0;
                                unsigned int iyMax = 0;
                */
                const unsigned int ixStart = (ox > 0) ? gridX[ox - 1] : 0;
                const unsigned int iyStart = (oy > 0) ? gridY[oy - 1] : 0;
                unsigned int ixStop = gridX[ox];
                unsigned int iyStop = gridY[oy];

                if (!overlapping) {
                    --ixStop;
                    --iyStop;
                }

                if (ox == outputsWidth - 1)
                    ixStop = channelsWidth - 1;

                if (oy == outputsHeight - 1)
                    iyStop = channelsHeight - 1;

                for (unsigned int iy = iyStart; iy <= iyStop; ++iy) {
                    for (unsigned int ix = ixStart; ix <= ixStop; ++ix) {
                        const unsigned int inputsIdx
                            = ix + (iy + output * channelsHeight)
                                   * channelsWidth;

                        if (inputs[inputsIdx + batchInputOffset] > poolValue) {
                            poolValue = inputs[inputsIdx + batchInputOffset];
                            /*
                                                        channelMax = channel;
                                                        ixMax = ix;
                                                        iyMax = iy;
                            */
                        }
                    }
                }

                // Compute the output signal
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth;
                outputs[outputsIdx + batchOutputOffset] = poolValue;
            }
        }
    }
}

void cudaSFMPPropagate(const DATA_T* inputs,
                       unsigned int* gridX,
                       unsigned int* gridY,
                       DATA_T* outputs,
                       unsigned int nbChannels,
                       unsigned int channelsHeight,
                       unsigned int channelsWidth,
                       unsigned int nbOutputs,
                       unsigned int outputsHeight,
                       unsigned int outputsWidth,
                       unsigned int batchSize,
                       bool overlapping)
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

    const uint3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const uint3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSFMPPropagate_kernel << <blocksPerGrid, threadsPerBlocks>>
        > (inputs,
           gridX,
           gridY,
           outputs,
           nbChannels,
           channelsHeight,
           channelsWidth,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           overlapping);
}
