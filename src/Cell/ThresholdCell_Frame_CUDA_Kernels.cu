/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include "Cell/ThresholdCell_Frame_CUDA_Kernels.hpp"

__global__
void cudaSThreshold_kernel(float* inputs,
                           unsigned int dimX,
                           unsigned int dimY,
                           unsigned int dimZ,
                           unsigned int batchSize,
                           float* outputs,
                           double threshold,
                           N2D2::ThresholdCell::Operation operation,
                           double maxValue)
{
    const unsigned int batchOffset = blockIdx.z * dimZ * dimY * dimX;

    for (unsigned int z = blockIdx.x; z < dimZ; z += gridDim.x) {
        for (unsigned int y = threadIdx.y; y < dimY; y += blockDim.y) {
            for (unsigned int x = threadIdx.x; x < dimX; x += blockDim.x) {
                const unsigned int idx
                    = x + dimX * (y + dimY * z) + batchOffset;
                
                if (operation == N2D2::ThresholdCell::Binary) {
                    outputs[idx]
                        = (inputs[idx] > threshold) ? maxValue : 0.0f;
                }
                else if (operation == N2D2::ThresholdCell::BinaryInverted) {
                    outputs[idx]
                        = (inputs[idx] > threshold) ? 0.0f : maxValue;
                }
                else if (operation == N2D2::ThresholdCell::Truncate) {
                    outputs[idx]
                        = (inputs[idx] > threshold) ? threshold : inputs[idx];
                }
                else if (operation == N2D2::ThresholdCell::ToZero) {
                    outputs[idx]
                        = (inputs[idx] > threshold) ? inputs[idx] : 0.0f;
                }
                else if (operation == N2D2::ThresholdCell::ToZeroInverted) {
                    outputs[idx]
                        = (inputs[idx] > threshold) ? 0.0f : inputs[idx];
                }
            }
        }
    }
}

void N2D2::cudaSThreshold(const cudaDeviceProp& deviceProp,
                          float* inputs,
                          unsigned int dimX,
                          unsigned int dimY,
                          unsigned int dimZ,
                          unsigned int batchSize,
                          float* outputs,
                          double threshold,
                          ThresholdCell::Operation operation,
                          double maxValue)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (dimX * dimY < maxSize)
                                       ? dimX * dimY
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)dimX);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {dimZ, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSThreshold_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (inputs,
           dimX,
           dimY,
           dimZ,
           batchSize,
           outputs,
           threshold,
           operation,
           maxValue);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
