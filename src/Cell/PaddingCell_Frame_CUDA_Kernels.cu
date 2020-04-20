
/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND(david.briand@cea.fr)

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

#include "Cell/PaddingCell_Frame_CUDA_Kernels.hpp"

__global__ void cudaSPadding_kernel( unsigned int nbOutputs,
                                     unsigned int outputWidth,
                                     unsigned int outputHeight,
                                     unsigned int nbChannels,
                                     unsigned int batchSize,
                                     unsigned int inputWidth,
                                     unsigned int inputHeight,
                                     int leftPad,
                                     int rightPad,
                                     int topPad,
                                     int botPad,
                                     const float* input,
                                     float* outputs)
{

    const unsigned int inputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*inputWidth*inputHeight; 

    const unsigned int outputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbOutputs*outputWidth*outputHeight;

    // nbCh = nbChannels for propagate
    //      = nbOutputs for back-propagate
    const unsigned int nbCh = min(nbChannels, nbOutputs);

    for (unsigned int ch = blockIdx.x; ch < nbCh; ch += gridDim.x) 
    {
        for (unsigned int oy = threadIdx.y; oy < outputHeight; oy += blockDim.y) 
        {
            for (unsigned int ox = threadIdx.x; ox < outputWidth; ox += blockDim.x) 
            {
                float outputValue = 0.0;
                int ix = (int) ox - leftPad;
                int iy = (int) oy - topPad;

                if( ix >= 0 && ix < (int) inputWidth
                    && iy >= 0 && iy < (int) inputHeight )
                {
                    outputValue = input[ix +  
                                        iy*inputWidth 
                                        + ch*inputWidth*inputHeight
                                        + inputOffset];

                }
                outputs[ ox + oy*outputWidth 
                         + ch*outputWidth*outputHeight + outputOffset]  = outputValue;

            }
        }
    }


}

void N2D2::cudaSPadding(const cudaDeviceProp& deviceProp,
                        unsigned int nbOutputs,
                        unsigned int outputsWidth,
                        unsigned int outputsHeight,
                        unsigned int nbChannels,
                        unsigned int batchSize,
                        unsigned int inputWidth,
                        unsigned int inputHeight,
                        int leftPad,
                        int rightPad,
                        int topPad,
                        int botPad,
                        const float* input,
                        float* outputs)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;

    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);
    const dim3 blocksPerGrid = {nbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSPadding_kernel<<<blocksPerGrid, threadsPerBlocks>>>( nbOutputs,
                                                            outputsWidth,
                                                            outputsHeight, 
                                                            nbChannels,
                                                            batchSize, 
                                                            inputWidth,
                                                            inputHeight, 
                                                            leftPad, 
                                                            rightPad, 
                                                            topPad, 
                                                            botPad, 
                                                            input,
                                                            outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}
