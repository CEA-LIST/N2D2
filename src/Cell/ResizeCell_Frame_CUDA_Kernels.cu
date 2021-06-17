
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

void N2D2::cudaSResizeFWBilinearTF(const cudaDeviceProp& deviceProp,
                                   unsigned int outputSizeX,
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
                                   float* outputs)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputSizeX * outputSizeY < maxSize)
                                       ? outputSizeX * outputSizeY
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputSizeX);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {outputNbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSBilinearTF_Forward_kernel<<<blocksPerGrid, threadsPerBlocks>>>( outputSizeX,
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

void N2D2::cudaSResizeBWBilinearTF(const cudaDeviceProp& deviceProp,
                           unsigned int outputSizeX,
                           unsigned int outputSizeY,
                           unsigned int outputNbChannels,
                           unsigned int batchSize,
                           unsigned int inputSizeX,
                           unsigned int inputSizeY,
                           const float scaleX,
                           const float scaleY,
                           const float* input,
                           float* outputs)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (inputSizeX * inputSizeY < maxSize)
                                       ? inputSizeX * inputSizeY
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)inputSizeX);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {outputNbChannels, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSBilinearTF_BackWard_kernel<<<blocksPerGrid, threadsPerBlocks>>>( outputSizeX,
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



__global__ void cudaSNearestNeighborKernel(const float* input, size_t inputSizeX, size_t inputSizeY,
                                            float* output, size_t outputSizeX, size_t outputSizeY,
                                            size_t nbChannels, size_t batchSize)
{
    const size_t inputOffset = (blockIdx.z*blockDim.z + threadIdx.z) * (nbChannels*inputSizeY*inputSizeX);
    const size_t outputOffset = (blockIdx.z*blockDim.z + threadIdx.z) * (nbChannels*outputSizeY*outputSizeX);
    
    const float multy = ((float) inputSizeY)/((float) outputSizeY);
    const float multx = ((float) inputSizeX)/((float) outputSizeX);

    for(size_t channel = blockIdx.x; channel < nbChannels; channel += gridDim.x) {
        for(size_t oy = threadIdx.y; oy < outputSizeY; oy += blockDim.y) {
            for(size_t ox = threadIdx.x; ox < outputSizeX; ox += blockDim.x) {
                const size_t iy = (size_t) oy*multy;
                const size_t ix = (size_t) ox*multx;


                output[outputOffset + 
                       channel*outputSizeY*outputSizeX +
                       oy*outputSizeX +
                       ox] = input[inputOffset + 
                                   channel*inputSizeY*inputSizeX +
                                   iy*inputSizeX +
                                   ix];
                                    
            }
        }
    }
}


void N2D2::cudaSResizeFWNearestNeighbor(const cudaDeviceProp& deviceProp,
                                        const float* input, size_t inputSizeX, size_t inputSizeY,
                                        float* output, size_t outputSizeX, size_t outputSizeY,
                                        size_t nbChannels, size_t batchSize) 
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputSizeX * outputSizeY < maxSize)
                                       ? outputSizeX * outputSizeY
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputSizeX);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {(unsigned int)nbChannels, 1, (unsigned int)batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSNearestNeighborKernel<<<blocksPerGrid, threadsPerBlocks>>>(input, inputSizeX, inputSizeY,
                                                                   output, outputSizeX, outputSizeY, 
                                                                   nbChannels, batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

void N2D2::cudaSResizeBWNearestNeighbor(const cudaDeviceProp& deviceProp,
                                        const float* input, size_t inputSizeX, size_t inputSizeY,
                                        float* output, size_t outputSizeX, size_t outputSizeY,
                                        size_t nbChannels, size_t batchSize) 
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (inputSizeX * inputSizeY < maxSize)
                                       ? inputSizeX * inputSizeY
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)inputSizeX);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {(unsigned int)nbChannels, 1, (unsigned int)batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSNearestNeighborKernel<<<blocksPerGrid, threadsPerBlocks>>>(input, inputSizeX, inputSizeY,
                                                                   output, outputSizeX, outputSizeY, 
                                                                   nbChannels, batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}