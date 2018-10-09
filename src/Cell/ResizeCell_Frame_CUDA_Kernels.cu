
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

__global__ void cudaSBilinearTF_Forward_kernel( unsigned int outputWidth,
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

__global__ void cudaSBilinearTF_BackWard_kernel( unsigned int outputWidth,
                                                unsigned int outputHeight,
                                                unsigned int nbChannels,
                                                unsigned int batchSize,
                                                unsigned int inputWidth,
                                                unsigned int inputHeight,
                                                const float scaleX,
                                                const float scaleY,
                                                const float* diffInput,
                                                float* diffOutputs)
{

    const unsigned int inputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*inputWidth*inputHeight;

    const unsigned int outputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*outputWidth*outputHeight;
    for (unsigned int ch = blockIdx.x; ch < nbChannels; ch += gridDim.x)
    {
        for (unsigned int oy = threadIdx.y; oy < outputHeight; oy += blockDim.y)
        {
            const float in_y = oy * scaleY;
            const int top_y_index = (int)(floorf(in_y));
            //const int bottom_y_index = min((int)(ceilf(in_y)), (int) (inputHeight - 1) ) ;

            const int bottom_y_index = (in_y < inputHeight - 1) ? ceilf(in_y) : inputHeight - 1;

            const float y_lerp = in_y - top_y_index;
            const float inverse_y_lerp = (1.0f - y_lerp);


            for (unsigned int ox = threadIdx.x; ox < outputWidth; ox += blockDim.x)
            {
                const float in_x = ox * scaleX;
                const int left_x_index = (int)(floorf(in_x));
                //const int right_x_index = min((int)(ceilf(in_x)), (int)(inputWidth - 1));
                const int right_x_index = (in_x < inputWidth - 1) ? ceilf(in_x) : inputWidth - 1;


                const float x_lerp = in_x - left_x_index;
                const float inverse_x_lerp = (1.0f - x_lerp);

                const unsigned int inLeftTopIdx = left_x_index + top_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;
                const unsigned int inRightTopIdx = right_x_index + top_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;
                const unsigned int inLeftBotIdx = left_x_index + bottom_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;
                const unsigned int inRightBotIdx = right_x_index + bottom_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;

                const unsigned int outIdx = ox + oy*outputWidth + ch*outputWidth*outputHeight + outputOffset;
                const float outData = diffInput[outIdx];

                diffOutputs[inLeftTopIdx]  += outData * inverse_y_lerp * inverse_x_lerp ;
                diffOutputs[inRightTopIdx]  += outData * inverse_y_lerp * x_lerp ;
                diffOutputs[inLeftBotIdx]  += outData * y_lerp * inverse_x_lerp ;
                diffOutputs[inRightBotIdx]  += outData * y_lerp * x_lerp ;

            }
        }
    }
}

void N2D2::cudaSResizeFWBilinearTF(unsigned int outputSizeX,
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

    cudaSBilinearTF_Forward_kernel<<<blocksPerGrid, threadsPerBlock>>>( outputSizeX,
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

void N2D2::cudaSResizeBWBilinearTF(unsigned int outputSizeX,
                           unsigned int outputSizeY,
                           unsigned int outputNbChannels,
                           unsigned int batchSize,
                           unsigned int inputSizeX,
                           unsigned int inputSizeY,
                           const float scaleX,
                           const float scaleY,
                           const float* input,
                           float* outputs,
                           const dim3 blocksPerGrid,
                           const dim3 threadsPerBlock)
{

    cudaSBilinearTF_BackWard_kernel<<<blocksPerGrid, threadsPerBlock>>>( outputSizeX,
                                                                        outputSizeY,
                                                                        outputNbChannels,
                                                                        batchSize,
                                                                        inputSizeX,
                                                                        inputSizeY,
                                                                        scaleX,
                                                                        scaleY,
                                                                        input,
                                                                        outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}
