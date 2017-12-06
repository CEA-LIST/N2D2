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

#include "Cell/FMPCell_Frame_CUDA_Kernels.hpp"

__global__ void cudaSFMPPropagate_kernel(float* inputs,
                                         unsigned int* gridX,
                                         unsigned int* gridY,
                                         float* outputs,
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

static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}

void N2D2::cudaSFMPPropagate(float* inputs,
                             unsigned int* gridX,
                             unsigned int* gridY,
                             float* outputs,
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

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSFMPPropagate_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (inputs,
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
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
