
/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Cell/RPCell_Frame_CUDA_Kernels.hpp"
#include <stdio.h>


__global__ void cudaSGatherRP_kernel( unsigned int inputSizeX,
                                        unsigned int inputSizeY,
                                        unsigned int nbAnchors,
                                        unsigned int batchSize,
                                        const float* inputs,
                                        const int* i,
                                        const int* j,
                                        const int* k,
                                        const int* b,
                                        const int* mask,
                                        float* outputs,
                                        int* anchors,
                                        unsigned int topN,
                                        const unsigned int nbProposals)
{
    const int batchPos = blockIdx.z;
    const int sortOffset = batchPos*topN;
    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;
    const int totalIndex = index + sortOffset;
    const int batchIndex = index + batchPos*nbProposals;

    if(index < nbProposals)
    {
        unsigned int xIdx = i[ mask[totalIndex] + sortOffset ] 
                             + j[ mask[totalIndex] + sortOffset ]*inputSizeX 
                             + (k[ mask[totalIndex] + sortOffset ] + nbAnchors)*inputSizeX*inputSizeY
                             + b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int yIdx = i[ mask[totalIndex] + sortOffset ] 
                            + j[ mask[totalIndex] + sortOffset ]*inputSizeX
                            + (k[ mask[totalIndex] + sortOffset ] + 2*nbAnchors)*inputSizeX*inputSizeY
                            + b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int wIdx = i[ mask[totalIndex] + sortOffset ] 
                            + j[ mask[totalIndex] + sortOffset ]*inputSizeX
                            + (k[ mask[totalIndex] + sortOffset ] + 3*nbAnchors)*inputSizeX*inputSizeY
                            + b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int hIdx = i[ mask[totalIndex] + sortOffset ] 
                            + j[ mask[totalIndex] + sortOffset ]*inputSizeX
                            + (k[ mask[totalIndex] + sortOffset ] + 4*nbAnchors)*inputSizeX*inputSizeY
                            + b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        anchors[0 + (batchIndex)*4] = i[mask[totalIndex]];
        anchors[1 + (batchIndex)*4] = j[mask[totalIndex]];
        anchors[2 + (batchIndex)*4] = k[mask[totalIndex]];
        anchors[3 + (batchIndex)*4] = b[mask[totalIndex]];

        outputs[0 + (batchIndex)*4] = inputs[xIdx];
        outputs[1 + (batchIndex)*4] = inputs[yIdx];
        outputs[2 + (batchIndex)*4] = inputs[wIdx];
        outputs[3 + (batchIndex)*4] = inputs[hIdx];                            
    }

}


__global__ void cudaSSplitIndexes_kernel( unsigned int inputSizeX,
                                        unsigned int inputSizeY,
                                        unsigned int nbAnchors, 
                                        unsigned int batchSize,
                                        const float* inputs, 
                                        float* values, 
                                        int* indexI, 
                                        int* indexJ, 
                                        int* indexK,
                                        int* indexB, 
                                        unsigned int* map, 
                                        float minWidth, 
                                        float minHeight, 
                                        unsigned int scoreIndex)
{
    const int batchPos = blockIdx.z;
    //const int batchInputOffset = batchPos*inputSizeX*inputSizeY*nbAnchors*(5 + scoreIndex);
    const int batchInputOffset = batchPos*inputSizeX*inputSizeY*nbAnchors*6;

    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;
    const int batchIndexOffset = batchPos*inputSizeX*inputSizeY*nbAnchors;

    if(index < inputSizeX*inputSizeY*nbAnchors)
    {
        float value = inputs[index + /*scoreIndex*inputSizeX*inputSizeY*nbAnchors +*/ batchInputOffset];      
        float w = inputs[index + 3*inputSizeX*inputSizeY*nbAnchors + batchInputOffset]; 
        float h = inputs[index + 4*inputSizeX*inputSizeY*nbAnchors + batchInputOffset];

        map[index + batchIndexOffset] = index;

        if(value >= 0.0 && w >= minWidth && h >= minHeight)
        {
            indexI[index + batchIndexOffset] = index%inputSizeX;
            indexJ[index + batchIndexOffset] = (index/inputSizeX)%inputSizeY;
            indexK[index + batchIndexOffset] = (index/(inputSizeX*inputSizeY))%nbAnchors;
            indexB[index + batchIndexOffset] = batchPos;
            values[index + batchIndexOffset] = value;         
        }
        else
            values[index  + batchIndexOffset] = -FLT_MAX;
        
    }  
}

__device__ inline float sIoU(const float x0, const float x,
                             const float y0, const float y,
                             const float w0, const float w,
                             const float h0, const float h) 
{
    float IoU = 0.0;
    const float interLeft = max(x0, x);
    const float interRight = min(x0 + w0, x + w);
    const float interTop = max(y0, y);
    const float interBottom = min(y0 + h0, y + h);
    
    if (interLeft < interRight && interTop < interBottom) {
        const float interArea = (interRight - interLeft)
                                    * (interBottom - interTop);
        const float unionArea = w0 * h0 + w * h - interArea;
        IoU = interArea / unionArea;
    }

    return IoU;
}




__global__ void cudaSnms_kernel( unsigned int inputSizeX,
                                 unsigned int inputSizeY,
                                 unsigned int nbAnchors,
                                 unsigned int batchSize,
                                 const float* inputs,
                                 const unsigned int inputOffset,
                                 int* i,
                                 int* j,
                                 int* k,
                                 int* b,
                                 const unsigned int indexOffset,
                                 unsigned long long* mask,
                                 const unsigned int outputOffset,
                                 const float nms_iou_thresh,
                                 const unsigned int max_nbBoxes,
                                 const unsigned int nbThreads)
{

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int row_size = min(max_nbBoxes - row * blockDim.x, blockDim.x);
    const int col_size = min(max_nbBoxes - col * blockDim.x, blockDim.x);

    __shared__ float shared_x0[64]; //(8*sizeof(unsigned long long) threads)
    __shared__ float shared_y0[64]; 
    __shared__ float shared_w0[64]; 
    __shared__ float shared_h0[64]; 
    
    if (threadIdx.x < col_size) {
        unsigned int x0Idx = i[col*64 + threadIdx.x + indexOffset] 
                            + j[col*64 + threadIdx.x + indexOffset]*inputSizeX 
                            + (k[col*64 + threadIdx.x + indexOffset] + nbAnchors)*inputSizeX*inputSizeY
                            + b[col*64 + threadIdx.x + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int y0Idx = i[col*64 + threadIdx.x + indexOffset] 
                            + j[col*64 + threadIdx.x + indexOffset]*inputSizeX
                            + (k[col*64 + threadIdx.x + indexOffset] + 2*nbAnchors)*inputSizeX*inputSizeY
                            + b[col*64 + threadIdx.x + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int w0Idx = i[col*64 + threadIdx.x + indexOffset] 
                            + j[col*64 + threadIdx.x + indexOffset]*inputSizeX
                            + (k[col*64 + threadIdx.x + indexOffset] + 3*nbAnchors)*inputSizeX*inputSizeY
                            + b[col*64 + threadIdx.x + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int h0Idx = i[col*64 + threadIdx.x + indexOffset] 
                            + j[col*64 + threadIdx.x + indexOffset]*inputSizeX
                            + (k[col*64 + threadIdx.x + indexOffset] + 4*nbAnchors)*inputSizeX*inputSizeY
                            + b[col*64 + threadIdx.x + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;
        //x0
        shared_x0[threadIdx.x] = inputs[x0Idx];
        //y0
        shared_y0[threadIdx.x] = inputs[y0Idx];
        //w0
        shared_w0[threadIdx.x] = inputs[w0Idx];
        //h0
        shared_h0[threadIdx.x] = inputs[h0Idx];
/*
        if(row == 0.0 && col == 0.0 && threadIdx.x == 0.0)
        {
            printf("[%d][%d][%d]{x0: %f y0: %f w0:%f h0: %f}", 
                                                    row, 
                                                    col,
                                                    threadIdx.x,
                                                    shared_x0[threadIdx.x], 
                                                    shared_y0[threadIdx.x], 
                                                    shared_w0[threadIdx.x],
                                                    shared_h0[threadIdx.x]);

            printf(" with {x0Idx: %d y0Idx: %d w0Idx:%d h0Idx: %d}", x0Idx, y0Idx, w0Idx, h0Idx);
            printf(" with {i: %d j: %d k:%d b: %d}\n",  i[col*64 + threadIdx.x + indexOffset],
                                                        j[col*64 + threadIdx.x + indexOffset],
                                                        k[col*64 + threadIdx.x + indexOffset],
                                                        b[col*64 + threadIdx.x + indexOffset]);
        }
*/
    }
    __syncthreads();

    if (threadIdx.x < row_size) 
    {
        const int cur_box_idx = blockDim.x * row + threadIdx.x;

        int boxIdx = 0;
        unsigned long long t = 0;
        int start = 0;

        if (row == col)
            start = threadIdx.x + 1;
    
        for (boxIdx = start; boxIdx < col_size; boxIdx++) {
            unsigned int xIdx = i[cur_box_idx + indexOffset] 
                                + j[cur_box_idx + indexOffset]*inputSizeX 
                                + (k[cur_box_idx + indexOffset] + nbAnchors)*inputSizeX*inputSizeY
                                + b[cur_box_idx + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;

            unsigned int yIdx = i[cur_box_idx + indexOffset] 
                                + j[cur_box_idx + indexOffset]*inputSizeX
                                + (k[cur_box_idx + indexOffset] + 2*nbAnchors)*inputSizeX*inputSizeY
                                + b[cur_box_idx + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;

            unsigned int wIdx = i[cur_box_idx + indexOffset] 
                                + j[cur_box_idx + indexOffset]*inputSizeX
                                + (k[cur_box_idx + indexOffset] + 3*nbAnchors)*inputSizeX*inputSizeY
                                + b[cur_box_idx + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;

            unsigned int hIdx = i[cur_box_idx + indexOffset] 
                                + j[cur_box_idx + indexOffset]*inputSizeX
                                + (k[cur_box_idx + indexOffset] + 4*nbAnchors)*inputSizeX*inputSizeY
                                + b[cur_box_idx + indexOffset]*nbAnchors*inputSizeX*inputSizeY*6;

            float IoU = sIoU(shared_x0[boxIdx], inputs[xIdx],
                             shared_y0[boxIdx], inputs[yIdx],
                             shared_w0[boxIdx], inputs[wIdx],
                             shared_h0[boxIdx], inputs[hIdx]);
                               
            if ( IoU > nms_iou_thresh) 
                t |= 1ULL << boxIdx;
            
        }

        const int col_blocks = DIVUP(max_nbBoxes, blockDim.x);

        mask[cur_box_idx * col_blocks + col + outputOffset] = t;
    }    
}


void N2D2::cudaSGatherRP( unsigned int inputSizeX,
                          unsigned int inputSizeY,
                          unsigned int nbAnchors,
                          unsigned int batchSize,
                          const float* inputs,
                          const int* i,
                          const int* j,
                          const int* k,
                          const int* b,
                          const int* mask,
                          float* outputs,
                          int* anchors,
                          unsigned int topN,
                          const unsigned int nbProposals,
                          const unsigned int nbBlocks)
{
    cudaSGatherRP_kernel<<<{nbBlocks,1, batchSize}, 32>>>( inputSizeX,
                                                            inputSizeY, 
                                                            nbAnchors,
                                                            batchSize, 
                                                            inputs, 
                                                            i, 
                                                            j, 
                                                            k, 
                                                            b, 
                                                            mask,
                                                            outputs,
                                                            anchors, 
                                                            topN,
                                                            nbProposals);

}

void N2D2::cudaSSplitIndexes(unsigned int inputSizeX,
                             unsigned int inputSizeY,
                             unsigned int nbAnchors,
                             unsigned int batchSize,
                             unsigned int nbBlocks,
                             const float* inputs,
                             float* values,
                             int* indexI,
                             int* indexJ,
                             int* indexK,
                             int* indexB,
                             unsigned int* map,
                             float minWidth,
                             float minHeight,
                             unsigned int scoreIndex)
{
    cudaSSplitIndexes_kernel<<<{nbBlocks, 1, batchSize}, 32>>>( inputSizeX,
                                                             inputSizeY, 
                                                             nbAnchors,
                                                             batchSize, 
                                                             inputs, 
                                                             values, 
                                                             indexI, 
                                                             indexJ, 
                                                             indexK, 
                                                             indexB, 
                                                             map,
                                                             minWidth, 
                                                             minHeight, 
                                                             scoreIndex);
}

void N2D2::cudaSnms( unsigned int inputSizeX,
                     unsigned int inputSizeY,
                     unsigned int nbAnchors,
                     unsigned int batchSize,
                     const float* inputs,
                     const unsigned int inputOffset,
                     int* i,
                     int* j,
                     int* k,
                     int* b,
                     const unsigned int indexOffset,
                     unsigned long long* mask,
                     const unsigned int outputOffset,
                     const float nms_iou_thresh,
                     const unsigned int max_nbBoxes,                     
                     const dim3 threadsPerBlock,
                     const dim3 blocksPerGrid)
{

    cudaSnms_kernel<<<blocksPerGrid, threadsPerBlock>>>( inputSizeX,
                                                         inputSizeY, 
                                                         nbAnchors,
                                                         batchSize, 
                                                         inputs, 
                                                         inputOffset,
                                                         i, 
                                                         j, 
                                                         k, 
                                                         b, 
                                                         indexOffset,
                                                         mask,
                                                         outputOffset,
                                                         nms_iou_thresh,
                                                         max_nbBoxes,
                                                         threadsPerBlock.x);

}

void N2D2::thrust_sort(float* inputs, unsigned int nbElements)
{

    const thrust::device_ptr<float> thrust_data(inputs);
    
    thrust::sort(thrust_data, thrust_data + nbElements, thrust::greater<float>());

}

void N2D2::thrust_sort_keys(float* inputs, unsigned int* keys, unsigned int nbElements, unsigned int offset)
{

    const thrust::device_ptr<float> thrust_data(inputs);
    const thrust::device_ptr<unsigned int> thrust_keys(keys);

    thrust::sort_by_key(thrust_data + offset, 
                        thrust_data + offset + nbElements, 
                        thrust_keys + offset, 
                        thrust::greater<float>());

}

void N2D2::thrust_gather(const unsigned int* keys, 
                         const int* inputs,
                         int* outputs, 
                         unsigned int nbElements,
                         unsigned int inputOffset,
                         unsigned int outputOffset)
{
    const thrust::device_ptr<const int> thrust_data_inputs(inputs);
    const thrust::device_ptr<const unsigned int> thrust_keys(keys);
    const thrust::device_ptr<int> thrust_data_outputs(outputs);
    thrust::gather(thrust_keys + inputOffset,
                   thrust_keys + inputOffset + nbElements,
                   thrust_data_inputs + inputOffset,
                   thrust_data_outputs + outputOffset);    

}
