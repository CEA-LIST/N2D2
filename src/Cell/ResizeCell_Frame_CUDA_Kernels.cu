
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

#include "Cell/ResizeCell_Frame_CUDA_Kernels.hpp"

__global__ void cudaSBilinearTF_kernel( unsigned int outputWidth,
                                        unsigned int outputHeight,
                                        unsigned int nbChannels,
                                        unsigned int batchSize,
                                        unsigned int inputWidth,
                                        unsigned int inputHeight,
                                        const unsigned int* yLowIdx,
                                        const unsigned int* yHighIdx,
                                        const float* yInter,
                                        const unsigned int* xLowIdx,
                                        const unsigned int* xHighIdx,
                                        const float* xInter,
                                        const float* input,
                                        float* outputs)
{

    const unsigned int inputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*inputWidth*inputHeight; 

    const unsigned int outputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*outputWidth*outputHeight;
    for (unsigned int ch = blockIdx.x; ch < nbChannels; ch += gridDim.x) 
    {
        for (unsigned int oy = threadIdx.y; oy < outputHeight; oy += blockDim.y) 
        {
            for (unsigned int ox = threadIdx.x; ox < outputWidth; ox += blockDim.x) 
            {
                const unsigned int indexTL = xLowIdx[ox] + yLowIdx[oy]*inputWidth 
                                            + ch*inputWidth*inputHeight
                                            + inputOffset;

                const unsigned int indexTR = xHighIdx[ox] + yLowIdx[oy]*inputWidth 
                                            + ch*inputWidth*inputHeight
                                            + inputOffset;

                const unsigned int indexBL = xLowIdx[ox] + yHighIdx[oy]*inputWidth 
                                            + ch*inputWidth*inputHeight
                                            + inputOffset;

                const unsigned int indexBR = xHighIdx[ox] + yHighIdx[oy]*inputWidth 
                                            + ch*inputWidth*inputHeight
                                            + inputOffset;

                const float top_left = input[indexTL];
                const float top_right = input[indexTR];
                const float bottom_left = input[indexBL];
                const float bottom_right = input[indexBR];

                const float top = top_left + (top_right - top_left) * xInter[ox];
                const float bottom = bottom_left + (bottom_right - bottom_left) * xInter[ox];

                outputs[ ox + oy*outputWidth 
                         + ch*outputWidth*outputHeight + outputOffset]  = top + (bottom - top) * yInter[oy];

            }
        }
    }
}

void N2D2::cudaSResizeBilinearTF(unsigned int outputSizeX,
                           unsigned int outputSizeY,
                           unsigned int outputNbChannels,
                           unsigned int batchSize,
                           unsigned int inputSizeX,
                           unsigned int inputSizeY,
                           unsigned int* yLowIdx,
                           unsigned int* yHighIdx,
                           float* yInter,
                           unsigned int* xLowIdx,
                           unsigned int* xHighIdx,
                           float* xInter,
                           const float* input,
                           float* outputs,
                           const dim3 blocksPerGrid,
                           const dim3 threadsPerBlock)
{

    cudaSBilinearTF_kernel<<<blocksPerGrid, threadsPerBlock>>>( outputSizeX,
                                                            outputSizeY, 
                                                            outputNbChannels,
                                                            batchSize, 
                                                            inputSizeX,
                                                            inputSizeY,
                                                            yLowIdx, 
                                                            yHighIdx, 
                                                            yInter, 
                                                            xLowIdx, 
                                                            xHighIdx,
                                                            xInter,
                                                            input,
                                                            outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}
