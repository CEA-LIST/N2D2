/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "common_cuda.hpp"
#include "kernels_gpu.hpp"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

//Utils
__device__ __inline__ float fclampf(float x, float min, float max)
{
    return (x < min) ? min : (x > max) ? max : x;
}

// Forward
__global__
void anchor_ca_kernel(unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        unsigned int stimuliHeight,
                        unsigned int stimuliWidth,
                        unsigned int scoresCls,
                        bool isFlip,
                        unsigned int nbAnchors,
                        const float* anchors,
                        const float* inputs,
                        float* outputs)
{
    const unsigned int batchInputOffset = blockIdx.z * (5 + scoresCls) * nbAnchors
                                        * outputHeight * outputWidth;
    const unsigned int batchOffset = blockIdx.z * 6 * nbAnchors
                                        * outputHeight * outputWidth;
    const float xOutputRatio = stimuliWidth;
    const float yOutputRatio = stimuliHeight;

    for (unsigned int k = blockIdx.x; k < nbAnchors;
         k += gridDim.x)
    {
        for (unsigned int ya = threadIdx.y; ya < outputHeight;
             ya += blockDim.y)
        {
            for (unsigned int xa = threadIdx.x; xa < outputWidth;
                 xa += blockDim.x)
            {
                const size_t anchorsPrecomputeIdx = xa*4 + ya*outputWidth*4 
                                                    + k*outputWidth*outputHeight*4;
                //Pre-computed anchors for each coordinates
                const float xac = anchors[anchorsPrecomputeIdx + 0];
                const float yac = anchors[anchorsPrecomputeIdx + 1];
                const float wa = anchors[anchorsPrecomputeIdx + 2];
                const float ha = anchors[anchorsPrecomputeIdx + 3];

                const unsigned int addrBase = batchOffset + xa + (ya + k * outputHeight) * outputWidth;
                const unsigned int addrStep = outputHeight * outputWidth;

                const unsigned int addrCoordBase = batchInputOffset 
                                                    + nbAnchors * addrStep
                                                    + k * outputWidth * outputHeight 
                                                    + ya * outputWidth 
                                                    + xa;

                const unsigned int addrClsBase = batchInputOffset 
                                                + k * outputWidth * outputHeight 
                                                + ya * outputWidth 
                                                + xa;

                // Score
                const float cls = inputs[addrClsBase];

                // Parameterized coordinates
                const float txbb = inputs[addrCoordBase + scoresCls * nbAnchors * addrStep];
                const float tybb = inputs[addrCoordBase + (scoresCls + 1) * nbAnchors * addrStep];
                const float twbb = inputs[addrCoordBase + (scoresCls + 2) * nbAnchors * addrStep];
                const float thbb = inputs[addrCoordBase + (scoresCls + 3) * nbAnchors * addrStep];

                // Predicted box center coordinates
                const float xbbc = ((isFlip) ? -txbb : txbb) * wa + xac;
                const float ybbc = ((isFlip) ? -tybb : tybb) * ha + yac;
                float wbb = wa * exp(twbb);
                float hbb = ha * exp(thbb);

                // Predicted box top-left coordinates
                float xbb = xbbc - wbb * 0.5;
                float ybb = ybbc - hbb * 0.5;

                /// During testing: "This  may  generate
                /// cross-boundary proposal boxes, which we clip to
                /// the image boundary."
                // Clip coordinates
                if (xbb < 0.0) {
                    wbb+= xbb;
                    xbb = 0.0;
                }
                if (ybb < 0.0) {
                    hbb+= ybb;
                    ybb = 0.0;
                }
                if (xbb + wbb > 1.0)
                    wbb = 1.0 - xbb;
                if (ybb + hbb > 1.0)
                    hbb = 1.0 - ybb;

                xbb *=  xOutputRatio;
                wbb *=  xOutputRatio;
                ybb *=  yOutputRatio;
                hbb *=  yOutputRatio;

                outputs[addrBase] = cls;
                outputs[addrBase + 1 * nbAnchors * addrStep] = xbb;
                outputs[addrBase + 2 * nbAnchors * addrStep] = ybb;
                outputs[addrBase + 3 * nbAnchors * addrStep] = wbb;
                outputs[addrBase + 4 * nbAnchors * addrStep] = hbb;
                outputs[addrBase + 5 * nbAnchors * addrStep] = 0.0;
            }
        }
    }
}
__global__
void anchor_ac_kernel(unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        unsigned int stimuliHeight,
                        unsigned int stimuliWidth,
                        unsigned int scoresCls,
                        bool isFlip,
                        unsigned int nbAnchors,
                        const float* anchors,
                        const float* inputs,
                        float* outputs)
{
    const unsigned int batchInputOffset = blockIdx.z * (5 + scoresCls) * nbAnchors
                                        * outputHeight * outputWidth;
    const unsigned int batchOffset = blockIdx.z * 6 * nbAnchors
                                        * outputHeight * outputWidth;
    const float xOutputRatio = stimuliWidth;
    const float yOutputRatio = stimuliHeight;

    for (unsigned int k = blockIdx.x; k < nbAnchors;
         k += gridDim.x)
    {
        for (unsigned int ya = threadIdx.y; ya < outputHeight;
             ya += blockDim.y)
        {
            for (unsigned int xa = threadIdx.x; xa < outputWidth;
                 xa += blockDim.x)
            {
                const size_t anchorsPrecomputeIdx = xa*4 + ya*outputWidth*4 
                                                    + k*outputWidth*outputHeight*4;
                //Pre-computed anchors for each coordinates
                const float xac = anchors[anchorsPrecomputeIdx + 0];
                const float yac = anchors[anchorsPrecomputeIdx + 1];
                const float wa = anchors[anchorsPrecomputeIdx + 2];
                const float ha = anchors[anchorsPrecomputeIdx + 3];

                const unsigned int addrBase = batchOffset + xa + (ya + k * outputHeight) * outputWidth;
                const unsigned int addrStep = outputHeight * outputWidth;

                const unsigned int addrCoordBase = batchInputOffset 
                                                    + nbAnchors * addrStep
                                                    + k * outputWidth * outputHeight *4
                                                    + ya * outputWidth 
                                                    + xa;

                const unsigned int addrClsBase = batchInputOffset 
                                                + k * outputWidth * outputHeight 
                                                + ya * outputWidth 
                                                + xa;

                // Score
                const float cls = inputs[addrClsBase];

                // Parameterized coordinates
                const float txbb = inputs[addrCoordBase + 0 * addrStep];
                const float tybb = inputs[addrCoordBase + 1 * addrStep];
                const float twbb = inputs[addrCoordBase + 2 * addrStep];
                const float thbb = inputs[addrCoordBase + 3 * addrStep];

                // Predicted box center coordinates
                const float xbbc = ((isFlip) ? -txbb : txbb) * wa + xac;
                const float ybbc = ((isFlip) ? -tybb : tybb) * ha + yac;
                float wbb = wa * exp(twbb);
                float hbb = ha * exp(thbb);

                // Predicted box top-left coordinates
                float xbb = xbbc - wbb * 0.5;
                float ybb = ybbc - hbb * 0.5;

                /// During testing: "This  may  generate
                /// cross-boundary proposal boxes, which we clip to
                /// the image boundary."
                // Clip coordinates
                if (xbb < 0.0) {
                    wbb+= xbb;
                    xbb = 0.0;
                }
                if (ybb < 0.0) {
                    hbb+= ybb;
                    ybb = 0.0;
                }
                if (xbb + wbb > 1.0)
                    wbb = 1.0 - xbb;
                if (ybb + hbb > 1.0)
                    hbb = 1.0 - ybb;

                xbb *=  xOutputRatio;
                wbb *=  xOutputRatio;
                ybb *=  yOutputRatio;
                hbb *=  yOutputRatio;

                outputs[addrBase] = cls;
                outputs[addrBase + 1 * nbAnchors * addrStep] = xbb;
                outputs[addrBase + 2 * nbAnchors * addrStep] = ybb;
                outputs[addrBase + 3 * nbAnchors * addrStep] = wbb;
                outputs[addrBase + 4 * nbAnchors * addrStep] = hbb;
                outputs[addrBase + 5 * nbAnchors * addrStep] = 0.0;
            }
        }
    }
}

__global__
void roipooling_bilinear_kernel(const float alpha,
                                const float* proposals,
                                unsigned int proposalIdx,
                                unsigned int nbProposals,
                                unsigned int inputSizeY,
                                unsigned int inputSizeX,
                                const float* inputs,
                                unsigned int nbChannels,
                                unsigned int channelsHeight,
                                unsigned int channelsWidth,
                                unsigned int batchSize,
                                unsigned int channelOffset,
                                const float beta,
                                float* outputs,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                unsigned int outputOffset,
                                bool bilinearTF,
                                bool isFlip,
                                bool ignorePadding)
{
    //const unsigned int proposalOffset = 4*proposalIdx;
    //const unsigned int batchProposalsOffset = proposalOffset + 4*nbProposals*blockIdx.z;
    const unsigned int batchProposalsOffset = 4*blockIdx.z;
    const unsigned int batchInputOffset = (blockIdx.z / nbProposals) * nbChannels * channelsHeight * channelsWidth;

    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;

    const float xRatio = ceilf(inputSizeX / (float)channelsWidth);
    const float yRatio = ceilf(inputSizeY / (float)channelsHeight);

    float xOffset = 0.0;
    float yOffset = 0.0;

    if (isFlip) {
        xOffset = (inputSizeX - 1) / xRatio
                    - (channelsWidth - 1);
        yOffset = (inputSizeY - 1) / yRatio
                    - (channelsHeight - 1);
    }


    float x = proposals[0 + batchProposalsOffset] / xRatio - xOffset;
    float y = proposals[1 + batchProposalsOffset] / yRatio - yOffset;
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

    float xPoolRatio, yPoolRatio;

    if (bilinearTF) {
        xPoolRatio = w / (outputsWidth - 1);
        yPoolRatio = h / (outputsHeight - 1);
    }
    else {
        xPoolRatio = w / outputsWidth;
        yPoolRatio = h / outputsHeight;
    }

    for (unsigned int channel = blockIdx.x; channel < nbChannels;
         channel += gridDim.x) {
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                 ox += blockDim.x)
            {
                float sx, sy;

                if (bilinearTF) {
                    sx = fminf(x + ox * xPoolRatio, channelsWidth - 1);
                    sy = fminf(y + oy * yPoolRatio, channelsHeight - 1);
                }
                else {
                    // -0.5 + (ox + 0.5) and not ox because the
                    // interpolation is done relative to the CENTER
                    // of the pixels
                    sx = x + fclampf( -0.5 + (ox + 0.5) * xPoolRatio, 0, w - 1);
                    sy = y + fclampf( -0.5 + (oy + 0.5) * yPoolRatio, 0, h - 1);
                }

                const unsigned int sx0 = (int)(sx);
                const unsigned int sy0 = (int)(sy);

                const float dx = sx - sx0;
                const float dy = sy - sy0;

                const unsigned int idxI00 = sx0 + sy0*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;

                const unsigned int idxI10 = (sx0 + 1) + sy0*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;

                const unsigned int idxI01 = sx0 + (sy0 + 1)*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;

                const unsigned int idxI11 = (sx0 + 1) + (sy0 + 1)*channelsWidth
                                            + channel*channelsHeight*channelsWidth
                                            + batchInputOffset;

                const bool invalid = ignorePadding ? (((sx0 + 1 < channelsWidth )  && (sy0 + 1 < channelsHeight ))  ? false : true) : false;

/**INITIAL
                const float i00 = inputs[idxI00];

                const float i10 = (sx0 + 1 < channelsWidth) ?
                                     inputs[idxI10] : 0.0;

                const float i01 = (sy0 + 1 < channelsHeight) ?
                                     inputs[idxI01]: 0.0;

                const float i11 = (sx0 + 1 < channelsWidth
                                     && sy0 + 1 < channelsHeight)
                                     ? inputs[idxI11] : 0.0;
**/
                const float i00 = (!invalid) ? inputs[idxI00] : 0.0;

                const float i10 = (sx0 + 1 < channelsWidth ) && (!invalid) ?
                                     inputs[idxI10] : 0.0;

                const float i01 = (sy0 + 1 < channelsHeight ) && (!invalid)  ?
                                     inputs[idxI01]: 0.0;

                const float i11 = (sx0 + 1 < channelsWidth
                                     && sy0 + 1 < channelsHeight ) && (!invalid)
                                     ? inputs[idxI11] : 0.0;


                const float value = i00 * (1 - dx) * (1 - dy)
                + i10 * dx * (1 - dy)
                + i01 * (1 - dx) * dy
                + i11 * (dx * dy);


                //const unsigned int outputsIdx
                //    = ox + (oy + channel*outputsHeight)
                //        * outputsWidth + outputOffset + batchOutputOffset;

                const unsigned int outputsIdx
                    = ox + (oy + (channel + outputOffset) * outputsHeight)
                        * outputsWidth + batchOutputOffset;

                //const float value =outputOffset;
                //outputs[outputsIdx] = alpha * value + beta * outputs[outputsIdx];

                outputs[outputsIdx] = alpha * value ;
            }
        }
    }
}

__global__ void batchnormcell_propagate_kernel( unsigned int nbChannels,
                                                unsigned int channelsHeight,
                                                unsigned int channelsWidth,
                                                const float* inputs,
                                                unsigned int nbOutputs_,
                                                unsigned int outputOffset,
                                                float* outputs,
                                                const float* bias,
                                                const float* variance,
                                                const float* mean,
                                                const float* scale,
                                                const float epsilon)
{
    const unsigned int batchOutputOffset
        = (blockIdx.z * blockDim.z + threadIdx.z)
          * nbOutputs_ * channelsWidth * channelsHeight; // Determine the output offset brings by batch size

    for (unsigned int output = blockIdx.x; output < nbOutputs_;
         output += gridDim.x) {
            const float var = sqrtf(variance[output] + epsilon);

        for (unsigned int oy = threadIdx.y; oy < channelsHeight;
             oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < channelsWidth;
                 ox += blockDim.x) {
                const unsigned int outputsIdx = ox + (oy + output*channelsHeight)*channelsWidth;

                const float normalized = ( (float) (inputs[outputsIdx + batchOutputOffset]) - mean[output]) / var ;
                const float sAs = scale[output]*normalized + bias[output]; //Scale and Shift normalized value


                outputs[batchOutputOffset + outputOffset + outputsIdx] = sAs;
            }
        }
    }
}
__global__ void cudaSGatherRP_kernel(   unsigned int inputSizeX,
                                        unsigned int inputSizeY,
                                        unsigned int nbAnchors,
                                        unsigned int batchSize,
                                        const float* inputs,
                                        const float* i,
                                        const float* j,
                                        const float* k,
                                        const float* b,
                                        const int* mask,
                                        float* outputs,
                                        const unsigned int topN,
                                        const unsigned int nbProposals)
{
    const int batchPos = blockIdx.z;
    const int sortOffset = batchPos*topN;

    int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    const int totalIndex = index + sortOffset;
    const int batchIndex = index + batchPos*nbProposals;

    if(index < nbProposals)
    {
        unsigned int xIdx = i[ mask[totalIndex] + sortOffset ]
                            + j[mask[totalIndex] + sortOffset ]*inputSizeX
                            + (k[mask[totalIndex] + sortOffset ] + nbAnchors)*inputSizeX*inputSizeY
                            + b[mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int yIdx = i[mask[totalIndex] + sortOffset ]
                            + j[mask[totalIndex] + sortOffset ]*inputSizeX
                            + (k[mask[totalIndex] + sortOffset ] + 2*nbAnchors)*inputSizeX*inputSizeY
                            + b[mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int wIdx = i[mask[totalIndex] + sortOffset ]
                            + j[mask[totalIndex] + sortOffset ]*inputSizeX
                            + (k[mask[totalIndex] + sortOffset ] + 3*nbAnchors)*inputSizeX*inputSizeY
                            + b[mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        unsigned int hIdx = i[mask[totalIndex] + sortOffset ]
                            + j[mask[totalIndex] + sortOffset ]*inputSizeX
                            + (k[mask[totalIndex] + sortOffset ] + 4*nbAnchors)*inputSizeX*inputSizeY
                            + b[mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

        outputs[0 + (batchIndex)*4] = inputs[xIdx];
        outputs[1 + (batchIndex)*4] = inputs[yIdx];
        outputs[2 + (batchIndex)*4] = inputs[wIdx];
        outputs[3 + (batchIndex)*4] = inputs[hIdx];
    }

}


__global__ void cudaSSplitIndexes_kernel(   unsigned int inputSizeX,
                                            unsigned int inputSizeY,
                                            unsigned int nbAnchors,
                                            unsigned int batchSize,
                                            const float* inputs,
                                            float* values,
                                            float* indexI,
                                            float* indexJ,
                                            float* indexK,
                                            float* indexB,
                                            unsigned int* map,
                                            float minWidth,
                                            float minHeight,
                                            unsigned int scoreIndex)
{
    int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    const int batchPos = blockIdx.z;
    //const int batchInputOffset = batchPos*inputSizeX*inputSizeY*nbAnchors*(5 + scoreIndex);
    const int batchInputOffset = batchPos*inputSizeX*inputSizeY*nbAnchors*6;
    const int batchIndexOffset = batchPos*inputSizeX*inputSizeY*nbAnchors;


    if(index < inputSizeX*inputSizeY*nbAnchors)
    {
        float value = inputs[index + /*scoreIndex*inputSizeX*inputSizeY*nbAnchors*/ + batchInputOffset];
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

__device__ inline float sIoU(   const float x0, const float x,
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




__global__ void cudaSnms_kernel(    unsigned int inputSizeX,
                                    unsigned int inputSizeY,
                                    unsigned int nbAnchors,
                                    unsigned int batchSize,
                                    const float* inputs,
                                    float* i,
                                    float* j,
                                    float* k,
                                    float* b,
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

            float IoU = sIoU(   shared_x0[boxIdx], inputs[xIdx],
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

__global__ void cudaSNormalizeROIs_kernel( unsigned int inputSizeX,
                                            unsigned int inputSizeY,
                                            unsigned int nbProposals,
                                            unsigned int batchSize,
                                            unsigned int scoreIdx,
                                            unsigned int nbCls,
                                            unsigned int maxParts,
                                            unsigned int maxTemplates,
                                            bool keepMax,
                                            bool generateParts,
                                            bool generateTemplates,
                                            const float normX,
                                            const float normY,
                                            const float* means,
                                            const float* std,
                                            const unsigned int* numPartsPerClass,
                                            const unsigned int* numTemplatesPerClass,
                                            const float* ROIRef,
                                            const float* ROIEst,
                                            const float* ValuesEst,
                                            const float* partsEst,
                                            const float* partsVisibilityEst,
                                            const float* templatesEst,
                                            float* outputs,
                                            int* argMax,
                                            float* partsPrediction,
                                            float* partsVisibilityPrediction,
                                            float* templatesPrediction,
                                            float scoreThreshold)
{
    const int batchPos = blockIdx.z*nbProposals;
    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    if(index < nbProposals)
    {

        unsigned int indexMin = scoreIdx;
        unsigned int indexMax = nbCls;

        if(keepMax)
        {
            unsigned int cls = scoreIdx;
            float maxVal = 0.0;

            for(unsigned int i = indexMin; i < indexMax; ++i)
            {
                unsigned int inputIdx = i + index*nbCls + batchPos*nbCls;

                if (ValuesEst[inputIdx] >= maxVal)
                {
                    maxVal = ValuesEst[inputIdx];
                    cls = i;
                }

            }
            argMax[index + batchPos] = cls;

            indexMin = cls;
            indexMax = cls + 1;
        }
        else
           argMax[index + batchPos] = -1;

        for(unsigned int clsIdx = indexMin; clsIdx < indexMax; ++clsIdx)
        {

            unsigned int bboxRefIdx = index*4 + batchPos*4;
            unsigned int bboxEstIdx = clsIdx*4 + index*4*nbCls + batchPos*4*nbCls;
            unsigned int valEstIdx = clsIdx + index*nbCls + batchPos*nbCls;

            unsigned int outputIdx = keepMax ? index*4*(nbCls - scoreIdx) + batchPos*4*(nbCls - scoreIdx)
                                        : (clsIdx - scoreIdx)*4 + index*4*(nbCls - scoreIdx) + batchPos*4*(nbCls - scoreIdx);


            const float xbbRef = ROIRef[0 + bboxRefIdx]*normX;
            const float ybbRef = ROIRef[1 + bboxRefIdx]*normY;
            const float wbbRef = ROIRef[2 + bboxRefIdx]*normX;
            const float hbbRef = ROIRef[3 + bboxRefIdx]*normY;


            const float xbbEst = ROIEst[0 + bboxEstIdx]*std[0] + means[0];

            const float ybbEst = ROIEst[1 + bboxEstIdx]*std[1] + means[1];

            const float wbbEst = ROIEst[2 + bboxEstIdx]*std[2] + means[2];

            const float hbbEst = ROIEst[3 + bboxEstIdx]*std[3] + means[3];


            float x = xbbEst*wbbRef + xbbRef + wbbRef/2.0 - (wbbRef/2.0)*exp(wbbEst);
            float y = ybbEst*hbbRef + ybbRef + hbbRef/2.0 - (hbbRef/2.0)*exp(hbbEst);
            float w = wbbRef*exp(wbbEst);
            float h = hbbRef*exp(hbbEst);

            /**Clip values**/
            if(x < 0.0)
            {
                w += x;
                x = 0.0;
            }

            if(y < 0.0)
            {
                h += y;
                y = 0.0;
            }

            w = ((w + x) > 1.0) ? (1.0 - x) / normX : w / normX;
            h = ((h + y) > 1.0) ? (1.0 - y) / normY : h / normY;

            x /= normX;
            y /= normY;

            if(ValuesEst[valEstIdx] >= scoreThreshold)
            {
                outputs[0 + outputIdx] = x;
                outputs[1 + outputIdx] = y;
                outputs[2 + outputIdx] = w;
                outputs[3 + outputIdx] = h;

                if(generateParts)
                {
                    unsigned int partsIdx = 0;
                    unsigned int proposalPartIdx = 0;
                    for(int idxCls = 0; idxCls < clsIdx; ++idxCls)
                        partsIdx += numPartsPerClass[idxCls];

                    proposalPartIdx = partsIdx;

                    for(int idxCls = clsIdx; idxCls < nbCls; ++idxCls)
                        proposalPartIdx += numPartsPerClass[idxCls];

                    for(unsigned int part = 0; part < numPartsPerClass[clsIdx];
                            ++part)
                    {
                        //const unsigned int partIdx = (partsIdx + part)*2;
                        const unsigned int inPartIdx = batchPos*2*proposalPartIdx
                                                        + index*2*proposalPartIdx
                                                        + partsIdx*2
                                                        + part*2;

                        const unsigned int outPartIdx = batchPos*2*maxParts*nbCls
                                                        + index*2*maxParts*nbCls
                                                        + clsIdx*2*maxParts
                                                        + part*2;

                        const float partY = partsEst[0 + inPartIdx];
                        const float partX = partsEst[1 + inPartIdx];
                        partsPrediction[0 + outPartIdx] = ((partY + 0.5) * hbbRef + ybbRef) / normY;
                        partsPrediction[1 + outPartIdx] = ((partX + 0.5) * wbbRef + xbbRef) / normX;

                        /// PARTS VISIBILITY PROCESSING
                        const unsigned int inPartVisibilityIdx = batchPos*4*proposalPartIdx
                                                                    + index*4*proposalPartIdx
                                                                    + partsIdx
                                                                    + part;

                        const unsigned int outPartVisibilityIdx = batchPos*maxParts*nbCls
                                                                    + index*maxParts*nbCls
                                                                    + clsIdx*maxParts
                                                                    + part;

                        float idxMax = 0.0;
                        float valueMax = partsVisibilityEst[inPartVisibilityIdx];

                        for(unsigned int v = 1; v < 4; ++v)
                        {
                            if(partsVisibilityEst[v*proposalPartIdx + inPartVisibilityIdx] >
                                        valueMax)
                            {
                                idxMax = (float) v;
                                valueMax = partsVisibilityEst[v*proposalPartIdx + inPartVisibilityIdx];
                            }
                        }
                        partsVisibilityPrediction[outPartVisibilityIdx] = idxMax;
                    }
                }


                if(generateTemplates)
                {
                    unsigned int templatesIdx = 0;
                    unsigned int proposalTemplateIdx = 0;
                    for(int idxCls = 0; idxCls < clsIdx; ++idxCls)
                        templatesIdx += numTemplatesPerClass[idxCls];

                    proposalTemplateIdx = templatesIdx;

                    for(int idxCls = clsIdx; idxCls < nbCls; ++idxCls)
                        proposalTemplateIdx += numTemplatesPerClass[idxCls];

                    for(unsigned int tpl = 0; tpl < numTemplatesPerClass[clsIdx];
                            ++tpl)
                    {
                        const unsigned int inTemplateIdx = batchPos*3*proposalTemplateIdx
                                                        + index*3*proposalTemplateIdx
                                                        + templatesIdx*3
                                                        + tpl*3;

                        const unsigned int outTemplateIdx = batchPos*3*maxTemplates*nbCls
                                                        + index*3*maxTemplates*nbCls
                                                        + clsIdx*3*maxTemplates
                                                        + tpl*3;

                        const float templateA = expf(templatesEst[0 + inTemplateIdx]);
                        const float templateB = expf(templatesEst[1 + inTemplateIdx]);
                        const float templateC = expf(templatesEst[2 + inTemplateIdx]);

                        templatesPrediction[0 + outTemplateIdx] = templateA;
                        templatesPrediction[1 + outTemplateIdx] = templateB;
                        templatesPrediction[2 + outTemplateIdx] = templateC;
                    }
                }
            }
            else
            {
                outputs[0 + outputIdx] = 0.0;
                outputs[1 + outputIdx] = 0.0;
                outputs[2 + outputIdx] = 0.0;
                outputs[3 + outputIdx] = 0.0;
            }
        }
    }

}


__global__ void cudaSToOutput_kernel( unsigned int nbProposals,
                                      const unsigned int scoreIdx,
                                      const unsigned int nbCls,
                                      const unsigned int nbOutputs,
                                      const unsigned int maxParts,
                                      const unsigned int maxTemplates,
                                      bool generateParts,
                                      bool generateTemplates,
                                      const unsigned int* numPartsPerClass,
                                      const unsigned int* numTemplatesPerClass,
                                      const int* maxCls,
                                      const float* inputs,
                                      const int* predictionIndex,
                                      const float* partsPrediction,
                                      const float* partsVisibilityPrediction,
                                      const float* templatesPrediction,
                                      float* outputs)
{
    const int batchPos = blockIdx.z*nbProposals;
    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    if(index < nbProposals)
    {
        const unsigned int inputIdx = index*4*(nbCls - scoreIdx)
                                        + batchPos*4*(nbCls - scoreIdx);

        //const unsigned int outputIdx = (nbOutputs == 4) ?
        //                                index*4 + batchPos*4
        //                                : index*5 + batchPos*5;
        unsigned int outputIdx = 0;
        unsigned offset = 0;

        if((nbOutputs == 4))
            outputIdx = index*4 + batchPos*4;
        else if((nbOutputs == 5))
            outputIdx = index*5 + batchPos*5;
        else if(generateParts && generateTemplates)
            outputIdx = (index + batchPos)*(5 + maxParts*3 + maxTemplates*3);
        else if(generateTemplates)
            outputIdx = (index + batchPos)*(5 + maxTemplates*3);
        else if(generateParts)
            outputIdx = (index + batchPos)*(5 + maxParts*3);


        outputs[0 + outputIdx] = inputs[0 + inputIdx];
        outputs[1 + outputIdx] = inputs[1 + inputIdx];
        outputs[2 + outputIdx] = inputs[2 + inputIdx];
        outputs[3 + outputIdx] = inputs[3 + inputIdx];

        offset = 4;

        if(nbOutputs > 4)
        {
            int cls = maxCls[index + batchPos];
            outputs[4 + outputIdx] = cls > -1 ?
                                    (float) cls
                                    : 0.0;
            offset += 1;
        }

        if(generateParts)
        {
            const int predProp = predictionIndex[(index + batchPos)*2 + 0];
            const int predCls = predictionIndex[(index + batchPos)*2 + 1];
            // PARTS PROCESSING
            if(predCls > -1)
            {
                for(unsigned int part = 0; part < numPartsPerClass[predCls];
                     ++part)
                {
                    const unsigned int partIdx = batchPos*maxParts*2*nbCls
                                            + predProp*maxParts*2*nbCls
                                            + predCls*maxParts*2
                                            + part*2;
                    outputs[0 + offset + part*2 + outputIdx] = partsPrediction[0 + partIdx];
                    outputs[1 + offset + part*2 + outputIdx] = partsPrediction[1 + partIdx];

                }
                for(int idx = numPartsPerClass[predCls]; idx < maxParts; ++idx)
                {
                    outputs[0 + offset + numPartsPerClass[predCls]*2 + idx*2 + outputIdx] = 0.0;
                    outputs[1 + offset + numPartsPerClass[predCls]*2 + idx*2 + outputIdx] = 0.0;
                }
            }
            offset += maxParts*2;

            if(predCls > -1)
            {
                // PARTS VISIBILITY PROCESSING
                for(unsigned int part = 0; part < numPartsPerClass[predCls];
                     ++part)
                {
                    const unsigned int partVisibilityIdx = batchPos*maxParts*nbCls
                                                            + predProp*maxParts*nbCls
                                                            + predCls*maxParts
                                                            + part;
                    outputs[offset + part + outputIdx] = partsVisibilityPrediction[partVisibilityIdx];

                }

                for(int idx = numPartsPerClass[predCls]; idx < maxParts; ++idx)
                    outputs[offset + numPartsPerClass[predCls] + idx + outputIdx] = -1.0;
            }
            offset += maxParts;
        }

        if(generateTemplates)
        {

            const int predProp = predictionIndex[(index + batchPos)*2 + 0];
            const int predCls = predictionIndex[(index + batchPos)*2 + 1];

            if(predCls > -1)
            {
                for(unsigned int tpl = 0; tpl < numTemplatesPerClass[predCls]; ++tpl)
                {
                    unsigned int templateIdx = batchPos*maxTemplates*3*nbCls
                                                + predProp*maxTemplates*3*nbCls
                                                + predCls*maxTemplates*3
                                                + tpl*3;

                    outputs[0 + offset + tpl*3 + outputIdx] = templatesPrediction[0 + templateIdx];
                    outputs[1 + offset + tpl*3 + outputIdx] = templatesPrediction[1 + templateIdx];
                    outputs[2 + offset + tpl*3 + outputIdx] = templatesPrediction[2 + templateIdx];

                }
                for(int idx = numTemplatesPerClass[predCls]; idx < maxParts; ++idx)
                {
                    outputs[0 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;
                    outputs[1 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;
                    outputs[2 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;

                }

            }
        }

    }
}

__global__ void spatial_output_kernel(unsigned int nbClass,
                                      unsigned int targetHeight,
                                      unsigned int targetWidth,
                                      float threshold,
                                      float* targetData,
                                      uint32_t* outputEstimated)
{
    const int batchInputOffset = targetWidth * targetHeight * nbClass * blockIdx.z;
    const int batchOutputOffset = targetWidth * targetHeight * blockIdx.z;

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < targetWidth * targetHeight; i += stride)
    {
        unsigned int outputMax = 0;

        if (nbClass > 1)
        {
                float maxVal = targetData[i + batchInputOffset];

                for (unsigned int cls = 1; cls < nbClass; ++cls) {
                    const float tmp = targetData[i + cls*targetWidth*targetHeight
                                            + batchInputOffset];

                    if (tmp > maxVal) {
                        outputMax = cls;
                        maxVal = tmp;
                    }
                }

                outputEstimated[i + batchOutputOffset] = outputMax;
        }
        else if(nbClass == 1)
        {
            if(targetData[index] > threshold)
                outputMax = 1;

            const int estimatedLabel
                = (targetData[i + batchInputOffset] > threshold);

            outputEstimated[i + batchOutputOffset] = estimatedLabel;

        }
    }
}

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



__global__ void cudaS_ssdToOutput_kernels(  unsigned int batchSize,
                                            unsigned int nbClass,
                                            unsigned int nbAnchors,
                                            unsigned int channelWidth,
                                            unsigned int channelHeight,
                                            unsigned int nbProposals,
                                            unsigned int* nbValidROIs,
                                            unsigned int cls,
                                            unsigned int totalParts,
                                            unsigned int totalTemplates,
                                            unsigned int maxParts,
                                            unsigned int maxTemplates,
                                            unsigned int cumulParts,
                                            unsigned int cumulTemplates,
                                            unsigned int nbParts,
                                            unsigned int nbTemplates,
                                            bool isCoordinatesAnchors,
                                            bool isPixelFormatXY,
                                            float xRatio,
                                            float yRatio,
                                            float xOutputRatio,
                                            float yOutputRatio,
                                            const float* roi_bbox,
                                            const float* roi_anchors,
                                            const float* anchors,
                                            const float* inputs_parts,
                                            const float* inputs_templates,
                                            float* outputs)
{
    const int batchPos = blockIdx.z;
    const int proposal = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;
    const int ptIdx = blockIdx.y;
    
    const int nbDetectedObject  = (int) nbValidROIs[batchPos];
    const int nbIdx = 6;

    if(proposal < nbProposals)
    {

        const unsigned int n = proposal + cls*nbProposals + batchPos*nbProposals*nbClass;

        if(proposal < nbDetectedObject)
        {
            if(ptIdx == 0)
            {
                outputs[0 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[0 + 5*proposal + batchPos*nbProposals*5];
                outputs[1 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[1 + 5*proposal + batchPos*nbProposals*5];
                outputs[2 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[2 + 5*proposal + batchPos*nbProposals*5];
                outputs[3 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[3 + 5*proposal + batchPos*nbProposals*5];
                outputs[4 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[4 + 5*proposal + batchPos*nbProposals*5];
                outputs[5 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = (float) cls;
            }
           
            if(ptIdx < nbParts && nbParts > 0)
            {
                const unsigned int xa   = roi_anchors[0 + 5*proposal + batchPos*nbProposals*5];
                const unsigned int ya   = roi_anchors[1 + 5*proposal + batchPos*nbProposals*5];
                const unsigned int k    = roi_anchors[2 + 5*proposal + batchPos*nbProposals*5];
                const unsigned int addBase = xa 
                                             + ya*channelWidth
                                             + cumulParts*channelHeight*channelWidth
                                             + batchPos*channelHeight*channelWidth*nbAnchors*2*totalParts;

                unsigned int xIdx = isPixelFormatXY ? addBase + ptIdx*2 * channelHeight*channelWidth
                                            : addBase + (ptIdx*2 + 1) * channelHeight*channelWidth;
                unsigned int yIdx = isPixelFormatXY ? addBase + (ptIdx*2 + 1) * channelHeight*channelWidth
                                            : addBase + ptIdx*2 * channelHeight*channelWidth;

                if(isCoordinatesAnchors) {
                    xIdx += k*nbParts*2*channelHeight*channelWidth;
                    yIdx += k*nbParts*2*channelHeight*channelWidth;
                } else {
                    xIdx += k*totalParts*2*channelHeight*channelWidth;
                    yIdx += k*totalParts*2*channelHeight*channelWidth;
                }


                const float partY = inputs_parts[yIdx];
                const float partX = inputs_parts[xIdx];

                const int xa0 = (int)(anchors[cls*nbAnchors*4 + k*4] + xa * xRatio);
                const int ya0 = (int)(anchors[cls*nbAnchors*4 + k*4 + 1] + ya * yRatio);
                const int xa1 = (int)(anchors[cls*nbAnchors*4 + k*4 + 2] + xa * xRatio);
                const int ya1 = (int)(anchors[cls*nbAnchors*4 + k*4 + 3] + ya * yRatio);

                // Anchors width and height
                const int wa = xa1 - xa0;
                const int ha = ya1 - ya0;

                // Anchor center coordinates (xac, yac)
                const float xac = xa0 + wa / 2.0;
                const float yac = ya0 + ha / 2.0;
                const float predPartY = ((partY) * ha + yac)*yOutputRatio ;
                const float predPartX = ((partX) * wa + xac)*xOutputRatio ;

                outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] 
                                                    = isPixelFormatXY ? predPartX : predPartY;
                outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] 
                                                    = isPixelFormatXY ? predPartY : predPartX;

            }
            else if(ptIdx < maxParts && maxParts > 0)
            {
                    outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                    outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }
            
            ///for(unsigned int t = 0; t < nbTemplates; ++t)
            if(ptIdx < nbTemplates && nbTemplates > 0)
            {
                const unsigned int xa   = roi_anchors[0 + 5*proposal + batchPos*nbProposals*5];
                const unsigned int ya   = roi_anchors[1 + 5*proposal + batchPos*nbProposals*5];
                const unsigned int k    = roi_anchors[2 + 5*proposal + batchPos*nbProposals*5];
                const unsigned int addBase = xa 
                                             + ya*channelWidth
                                             + cumulTemplates*channelHeight*channelWidth
                                             + batchPos*channelHeight*channelWidth*nbAnchors*3*totalTemplates;

                unsigned int xIdx = isPixelFormatXY ? addBase + ptIdx*3 * channelHeight*channelWidth
                                            : addBase + (ptIdx*3 + 1) * channelHeight*channelWidth;
                unsigned int yIdx = isPixelFormatXY ? addBase + (ptIdx*3 + 1) * channelHeight*channelWidth
                                            : addBase + ptIdx*3 * channelHeight*channelWidth;
                unsigned int zIdx =  addBase + (ptIdx*3 + 2) * channelHeight*channelWidth;

                if(isCoordinatesAnchors) {
                    xIdx += k*nbTemplates*3*channelHeight*channelWidth;
                    yIdx += k*nbTemplates*3*channelHeight*channelWidth;
                    zIdx += k*nbTemplates*3*channelHeight*channelWidth;

                } else {
                    xIdx += k*totalTemplates*3*channelHeight*channelWidth;
                    yIdx += k*totalTemplates*3*channelHeight*channelWidth;
                    zIdx += k*totalTemplates*3*channelHeight*channelWidth;
                }

                const float templateY = expf(inputs_templates[yIdx]);
                const float templateX = expf(inputs_templates[xIdx]);
                const float templateZ = expf(inputs_templates[zIdx]);

                outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] 
                                                    = isPixelFormatXY ? templateX : templateY;
                outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] 
                                                    = isPixelFormatXY ? templateY : templateX;
                outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = templateZ;

            }
            else if(ptIdx < maxTemplates && maxTemplates > 0)
            {
                    outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                    outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                    outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }
        }
        else
        {
            outputs[0 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[1 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[2 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[3 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            outputs[4 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
            //for(unsigned int p = 0; p < nbParts; ++p)
            if(ptIdx < maxParts && maxParts > 0)
            {

                outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }

            //for(unsigned int t = 0;t < nbTemplates; ++t)
            if(ptIdx < maxTemplates && maxTemplates > 0)
            {
                outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
                outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
            }
        }
    }
}



extern "C" void cudaS_SSD_output_gathering( unsigned int batchSize,
                                            unsigned int nbClass,
                                            unsigned int nbAnchors,
                                            unsigned int channelWidth,
                                            unsigned int channelHeight,
                                            unsigned int nbProposals,
                                            unsigned int* nbValidROIs,
                                            unsigned int cls,
                                            unsigned int totalParts,
                                            unsigned int totalTemplates,
                                            unsigned int maxParts,
                                            unsigned int maxTemplates,
                                            unsigned int cumulParts,
                                            unsigned int cumulTemplates,
                                            unsigned int nbParts,
                                            unsigned int nbTemplates,
                                            bool isCoordinatesAnchors,
                                            bool isPixelFormatXY,
                                            float xRatio,
                                            float yRatio,
                                            float xOutputRatio,
                                            float yOutputRatio,
                                            const float* roi_bbox,
                                            const float* roi_anchors,
                                            const float* anchors,
                                            const float* inputs_parts,
                                            const float* inputs_templates,
                                            float* outputs,
                                            const dim3 blocksPerGrid,
                                            const dim3 threadsPerBlock)
{

    cudaS_ssdToOutput_kernels<<<blocksPerGrid, threadsPerBlock>>>( batchSize,
                                                                nbClass,
                                                                nbAnchors,
                                                                channelWidth,
                                                                channelHeight,
                                                                nbProposals,
                                                                nbValidROIs,
                                                                cls,
                                                                totalParts,
                                                                totalTemplates,
                                                                maxParts,
                                                                maxTemplates,
                                                                cumulParts,
                                                                cumulTemplates,
                                                                nbParts,
                                                                nbTemplates,
                                                                isCoordinatesAnchors,
                                                                isPixelFormatXY,
                                                                xRatio,
                                                                yRatio,
                                                                xOutputRatio,
                                                                yOutputRatio,
                                                                roi_bbox,
                                                                roi_anchors,
                                                                anchors,
                                                                inputs_parts,
                                                                inputs_templates,
                                                                outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}

__global__ void cudaSReduceIndex_kernel(  const unsigned int inputSize,
                                          const unsigned int inputBatchOffset,
                                          const unsigned int outputBatchOffset,
                                          const unsigned int channelsWidth,
                                          const unsigned int channelsHeight,
                                          const unsigned int nbAnchors,
                                          const float* valueThreshold,
                                          const float* inputs,
                                          int* outputMap,
                                          float* scores)
{
    const int batchPos = blockIdx.z;
    const int clsPos = blockIdx.y;

    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    const int inputIndex = index 
                            + inputSize*blockIdx.y
                            + batchPos*inputBatchOffset;

    const int outputIndex = index
                            + inputSize*blockIdx.y
                            + batchPos*outputBatchOffset;

    if(index < inputSize)
    {
        float value = inputs[inputIndex];      

        if(value >= valueThreshold[clsPos])
        {
            outputMap[outputIndex] = index;
            scores[outputIndex] = value;
        }
        else
        {
            outputMap[outputIndex] = -255;
            scores[outputIndex] = -255.0;
        }
    }  
}

extern "C" void cudaSReduceIndex(  const unsigned int inputSize,
                                    const unsigned int inputBatchOffset,
                                    const unsigned int outputBatchOffset,
                                    const unsigned int channelWidth,
                                    const unsigned int channelHeight,
                                    const unsigned int nbAnchors,
                                    const float* valueThreshold,
                                    const float* inputs,
                                    int* outputMap,
                                    float* scores,
                                    const dim3 blocksPerGrid,
                                    const dim3 threadsPerBlock)
{

    cudaSReduceIndex_kernel<<<blocksPerGrid, threadsPerBlock>>>( inputSize,
                                                                 inputBatchOffset,
                                                                 outputBatchOffset,
                                                                 channelWidth,
                                                                 channelHeight,
                                                                 nbAnchors,
                                                                 valueThreshold, 
                                                                 inputs,
                                                                 outputMap,
                                                                 scores);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}

__global__ void cudaSgatherI2I_kernel( const int* keys,
                                        const int* indicesX,
                                        const int* indicesY,
                                        const int* indicesK,
                                         int* outX,
                                         int* outY,
                                         int* outK,
                                        unsigned int nbElements)
{
    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    if(index < nbElements)
    {
        const int key = keys[index];
        printf("keys[%d]=%d indicesX[%d]:%d  ", index, key, index, indicesX[index] );
        outX[index] = indicesX[key];
        outY[index] = indicesY[key];
        outK[index] = indicesK[key];
    }
}


extern "C" void cuda_gather_int2int_indices( const int* keys,
                                             const int* indicesX,
                                             const int* indicesY,
                                             const int* indicesK,
                                             int* outX,
                                             int* outY,
                                             int* outK,
                                             unsigned int nbElements,
                                             const dim3 blocksPerGrid,
                                             const dim3 threadsPerBlock)
{
    cudaSgatherI2I_kernel<<<blocksPerGrid, threadsPerBlock>>>( keys,
                                                                 indicesX,
                                                                 indicesY,
                                                                 indicesK,
                                                                 outX,
                                                                 outY,
                                                                 outK,
                                                                 nbElements);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
__constant__ static float colors[10][3] = {{255.0, 0.0, 0.0},{0.0, 255.0, 0.0},{0.0, 0.0, 255.0},{0.0, 0.0, 0.0},
                                          {0.0, 255.0, 255.0},{255.0, 128.0, 255.0},{255.0, 255.0, 0.0},{128.0, 0.0, 128.0},
                                          {0.0, 128.0, 128.0},{255.0, 50.0, 50.0}};

__global__ void add_weighted_kernel(unsigned int batchSize,
                                      unsigned int nbOutputs,
                                      unsigned int outputsHeight,
                                      unsigned int outputsWidth,
                                      float* estimated_labels,
                                      unsigned int nbChannels,
                                      unsigned int image_height,
                                      unsigned int image_width,
                                      float* input_image,
                                      unsigned char* workspace,
                                      float alpha)
{
    const int batchEstimatedOffset = nbOutputs * outputsHeight * outputsWidth * blockIdx.z;
    const int batchImageOffset = nbChannels * image_height * image_width * blockIdx.z;

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < outputsWidth * outputsHeight; i += stride)
    {
        unsigned int outputMax = 0;

        if (nbOutputs > 1)
        {
                float maxVal = estimated_labels[i + batchEstimatedOffset];

                for (unsigned int cls = 1; cls < nbOutputs; ++cls) {
                    const float tmp = estimated_labels[i 
                                                        + cls*outputsWidth*outputsHeight
                                                        + batchEstimatedOffset];

                    if (tmp > maxVal) {
                        outputMax = cls;
                        maxVal = tmp;
                    }
                }
                const unsigned char ch0 
                    = (unsigned char) max(colors[outputMax%10][0]*alpha, min(255.0, colors[outputMax%10][0]*alpha + input_image[i + batchImageOffset]));
                const unsigned char ch1 
                    = (unsigned char) max(colors[outputMax%10][1]*alpha, min(255.0, colors[outputMax%10][1]*alpha + input_image[i + image_height*image_width + batchImageOffset]));
                const unsigned char ch2 
                    = (unsigned char) max(colors[outputMax%10][2]*alpha, min(255.0, colors[outputMax%10][2]*alpha + input_image[i + 2*image_height*image_width + batchImageOffset]));

                workspace[i*3 + batchImageOffset] = ch0;
                workspace[i*3 + 1 + batchImageOffset] = ch1;
                workspace[i*3 + 2 + batchImageOffset] = ch2;
        }
    }
}

extern "C" void cuda_add_weighted(unsigned int batchSize,
                                  unsigned int nbOutputs,
                                  unsigned int outputsHeight,
                                  unsigned int outputsWidth,
                                  float* estimated_labels,
                                  unsigned int nbChannels,
                                  unsigned int image_height,
                                  unsigned int image_width,
                                  float* input_image,
                                  unsigned char* workspace,
                                  float alpha,
                                  const dim3 threadsPerBlock,
                                  const dim3 blocksPerGrid,
                                  cudaStream_t stream)
{
    add_weighted_kernel <<<blocksPerGrid, threadsPerBlock, 0, stream>>>
                            (batchSize,
                              nbOutputs,
                              outputsHeight,
                              outputsWidth,
                              estimated_labels,
                              nbChannels,
                              image_height,
                              image_width,
                              input_image,
                              workspace,
                              alpha);

}

extern "C" int copy_if_int(const int* inputs,
                                int* outputs, 
                                unsigned int nbElements)
{
    const thrust::device_ptr<const int> thrust_data_inputs(inputs);
    const thrust::device_ptr<int> thrust_data_outputs(outputs);
    thrust::device_ptr<int> return_ptr =  thrust::copy_if(  thrust_data_inputs,
                                                            thrust_data_inputs + nbElements,
                                                            thrust_data_outputs ,
                                                            thrust::placeholders::_1 > -1); 
    int nbCpyElements = (int) (return_ptr - thrust_data_outputs);

    return nbCpyElements;
}

extern "C" int copy_if_float(const float* inputs,
                                float* outputs, 
                                unsigned int nbElements)
{
    const thrust::device_ptr<const float> thrust_data_inputs(inputs);
    const thrust::device_ptr<float> thrust_data_outputs(outputs);
    thrust::device_ptr<float> return_ptr = thrust::copy_if(  thrust_data_inputs,
                                                            thrust_data_inputs + nbElements,
                                                            thrust_data_outputs ,
                                                            thrust::placeholders::_1 > -1.0);  

    int nbCpyElements = (int) (return_ptr - thrust_data_outputs);

    return nbCpyElements;
}


extern "C" void cuda_region_proposal_split_indexes( unsigned int inputSizeX,
                                                    unsigned int inputSizeY,
                                                    unsigned int nbAnchors,
                                                    unsigned int batchSize,
                                                    unsigned int nbBlocks,
                                                    const float* inputs,
                                                    float* values,
                                                    float* indexI,
                                                    float* indexJ,
                                                    float* indexK,
                                                    float* indexB,
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

extern "C" void cuda_region_proposal_nms(   unsigned int inputSizeX,
                                            unsigned int inputSizeY,
                                            unsigned int nbAnchors,
                                            unsigned int batchSize,
                                            const float* inputs,
                                            float* i,
                                            float* j,
                                            float* k,
                                            float* b,
                                            const unsigned int indexOffset,
                                            unsigned long long* mask,
                                            const unsigned int outputOffset,
                                            const float nms_iou_thresh,
                                            const unsigned int max_nbBoxes,
                                            const dim3 threadsPerBlock,
                                            const dim3 blocksPerGrid)
{

    cudaSnms_kernel<<<blocksPerGrid, threadsPerBlock>>>(inputSizeX,
                                                        inputSizeY,
                                                        nbAnchors,
                                                        batchSize,
                                                        inputs,
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


extern "C" void cuda_region_proposal_gathering( unsigned int inputSizeX,
                                                unsigned int inputSizeY,
                                                unsigned int nbAnchors,
                                                unsigned int batchSize,
                                                const float* inputs,
                                                const float* i,
                                                const float* j,
                                                const float* k,
                                                const float* b,
                                                const int* mask,
                                                float* outputs,
                                                const unsigned int topN,
                                                const unsigned int nbProposals,
                                                const unsigned int nbBlocks)
{
    cudaSGatherRP_kernel<<<{nbBlocks, 1, batchSize}, 32>>>( inputSizeX,
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
                                                           topN,
                                                           nbProposals);

}


extern "C" void cuda_proposal_normalize( unsigned int inputSizeX,
                                        unsigned int inputSizeY,
                                        unsigned int nbProposals,
                                        unsigned int batchSize,
                                        unsigned int scoreIdx,
                                        unsigned int nbCls,
                                        unsigned int maxParts,
                                        unsigned int maxTemplates,
                                        bool keepMax,
                                        bool generateParts,
                                        bool generateTemplates,
                                        const float normX,
                                        const float normY,
                                        const float* means,
                                        const float* std,
                                        const unsigned int* numPartsPerClass,
                                        const unsigned int* numTemplatesPerClass,
                                        const float* ROIRef,
                                        const float* ROIEst,
                                        const float* ValuesEst,
                                        const float* partsEst,
                                        const float* partsVisibilityEst,
                                        const float* templatesEst,
                                        float* outputs,
                                        int* argMax,
                                        float* partsPrediction,
                                        float* partsVisibilityPrediction,
                                        float* templatesPrediction,
                                        float scoreThreshold,
                                        const dim3 threadsPerBlock,
                                        const dim3 blocksPerGrid)
{

    cudaSNormalizeROIs_kernel<<<blocksPerGrid, threadsPerBlock>>>( inputSizeX,
                                                                   inputSizeY,
                                                                    nbProposals,
                                                                    batchSize,
                                                                    scoreIdx,
                                                                    nbCls,
                                                                    maxParts,
                                                                    maxTemplates,
                                                                    keepMax,
                                                                    generateParts,
                                                                    generateTemplates,
                                                                    normX,
                                                                    normY,
                                                                    means,
                                                                    std,
                                                                    numPartsPerClass,
                                                                    numTemplatesPerClass,
                                                                    ROIRef,
                                                                    ROIEst,
                                                                    ValuesEst,
                                                                    partsEst,
                                                                    partsVisibilityEst,
                                                                    templatesEst,
                                                                    outputs,
                                                                    argMax,
                                                                    partsPrediction,
                                                                    partsVisibilityPrediction,
                                                                    templatesPrediction,
                                                                    scoreThreshold);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

extern "C" void cuda_proposal_to_output( const unsigned int nbProposals,
                                         const unsigned int scoreIdx,
                                         const unsigned int nbCls,
                                         const unsigned int nbOutputs,
                                         unsigned int maxParts,
                                         unsigned int maxTemplates,
                                         bool generateParts,
                                         bool generateTemplates,
                                         const unsigned int* numPartsPerClass,
                                         const unsigned int* numTemplatesPerClass,
                                         const int* maxCls,
                                         const float* input,
                                         const int* predictionIndex,
                                         const float* partsPrediction,
                                         const float* partsVisibilityPrediction,
                                         const float* templatesPrediction,
                                         float* outputs,
                                         const dim3 threadsPerBlock,
                                         const dim3 blocksPerGrid)
{

    cudaSToOutput_kernel<<<blocksPerGrid, threadsPerBlock>>>( nbProposals,
                                                              scoreIdx,
                                                              nbCls,
                                                              nbOutputs,
                                                              maxParts,
                                                              maxTemplates,
                                                              generateParts,
                                                              generateTemplates,
                                                              numPartsPerClass,
                                                              numTemplatesPerClass,
                                                              maxCls,
                                                              input,
                                                              predictionIndex,
                                                              partsPrediction,
                                                              partsVisibilityPrediction,
                                                              templatesPrediction,
                                                              outputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());


}

extern "C" void cuda_resize_bilinearTF_propagate(   unsigned int outputSizeX,
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
                                                    const dim3 threadsPerBlock,
                                                    cudaStream_t stream)
{
    cudaSBilinearTF_kernel <<<blocksPerGrid, threadsPerBlock, 0, stream>>>
                                                                        (outputSizeX,
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

extern "C" void cuda_anchor_propagate(  unsigned int batchSize,
                                        unsigned int nbOutputs,
                                        unsigned int outputHeight,
                                        unsigned int outputWidth,
                                        unsigned int stimuliHeight,
                                        unsigned int stimuliWidth,
                                        unsigned int scoreCls,
                                        bool isCoordinatesAnchors,
                                        bool isFlip,
                                        unsigned int nbAnchors,
                                        const float* anchors,
                                        const float* inputs,
                                        float* outputs,
                                        dim3 threadsPerBlocks,
                                        dim3 blocksPerGrid,
                                        cudaStream_t stream)
{
    if(isCoordinatesAnchors) {
        anchor_ca_kernel <<<blocksPerGrid, threadsPerBlocks, 0, stream>>>
                            (batchSize,
                            nbOutputs,
                            outputHeight,
                            outputWidth,
                            stimuliHeight,
                            stimuliWidth,
                            scoreCls,
                            isFlip,
                            nbAnchors,
                            anchors,
                            inputs,
                            outputs);
    }
    else {
        anchor_ac_kernel <<<blocksPerGrid, threadsPerBlocks, 0, stream>>>
                            (batchSize,
                            nbOutputs,
                            outputHeight,
                            outputWidth,
                            stimuliHeight,
                            stimuliWidth,
                            scoreCls,
                            isFlip,
                            nbAnchors,
                            anchors,
                            inputs,
                            outputs);
    }

}

extern "C" void cuda_batchnormcell_propagate(unsigned int nbChannels,
                                            unsigned int channelsHeight,
                                            unsigned int channelsWidth,
                                            const float* inputs,
                                            unsigned int nbOutputs_,
                                            unsigned int outputOffset,
                                            float* outputs,
                                            const float* bias,
                                            const float* variance,
                                            const float* mean,
                                            const float* scale,
                                            const float epsilon,
                                            dim3 threadsPerBlocks,
                                            dim3 blocksPerGrid,
                                            cudaStream_t stream)

{

    batchnormcell_propagate_kernel <<<blocksPerGrid, threadsPerBlocks, 0, stream>>> (nbChannels,
           channelsHeight,
           channelsWidth,
           inputs,
           nbOutputs_,
           outputOffset,
           outputs,
           bias,
           variance,
           mean,
           scale,
           epsilon);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

}

extern "C" void cuda_roipooling_bilinear_propagate
                                                (const float alpha,
                                                 const float* proposals,
                                                 unsigned int proposalIdx,
                                                 unsigned int nbProposals,
                                                 unsigned int inputSizeY,
                                                 unsigned int inputSizeX,
                                                 const float* inputs,
                                                 unsigned int nbChannels,
                                                 unsigned int channelsHeight,
                                                 unsigned int channelsWidth,
                                                 unsigned int batchSize,
                                                 unsigned int channelOffset,
                                                 const float beta,
                                                 float* outputs,
                                                 unsigned int nbOutputs,
                                                 unsigned int outputsHeight,
                                                 unsigned int outputsWidth,
                                                 unsigned int outputOffset,
                                                 bool bilinearTF,
                                                 bool isFlip,
                                                 bool ignorePadding,
                                                 dim3 threadsPerBlocks,
                                                 dim3 blocksPerGrid,
                                                 cudaStream_t stream)
{
    roipooling_bilinear_kernel <<<blocksPerGrid, threadsPerBlocks, 0, stream>>>
                            (alpha,
                              proposals,
                              proposalIdx,
                              nbProposals,
                              inputSizeY,
                              inputSizeX,
                              inputs,
                              nbChannels,
                              channelsHeight,
                              channelsWidth,
                              batchSize,
                              channelOffset,
                              beta,
                              outputs,
                              nbOutputs,
                              outputsHeight,
                              outputsWidth,
                              outputOffset,
                              bilinearTF,
                              isFlip,
                              ignorePadding);

}

extern "C" void cuda_spatial_outputs(unsigned int nbClass,
                                         unsigned int targetHeight,
                                         unsigned int targetWidth,
                                         unsigned int batchSize,
                                         float threshold,
                                         float* targetData,
                                         uint32_t* outputEstimated,
                                         dim3 threadsPerBlocks,
                                         dim3 blocksPerGrid,
                                         cudaStream_t stream)
{
    spatial_output_kernel <<<blocksPerGrid, threadsPerBlocks, 0, stream>>>
                                                                        (nbClass,
                                                                        targetHeight,
                                                                        targetWidth,
                                                                        threshold,
                                                                        targetData,
                                                                        outputEstimated);

}


extern "C" void thrust_sort(float* inputs, unsigned int nbElements)
{

    const thrust::device_ptr<float> thrust_data(inputs);

    thrust::sort(thrust_data, thrust_data + nbElements, thrust::greater<float>());

}

extern "C" void thrust_sort_keys(float* inputs, unsigned int* keys, unsigned int nbElements,  unsigned int offset)
{

    const thrust::device_ptr<float> thrust_data(inputs);
    const thrust::device_ptr<unsigned int> thrust_keys(keys);

    thrust::stable_sort_by_key( thrust_data + offset,
                                thrust_data + offset + nbElements,
                                thrust_keys + offset,
                                thrust::greater<float>());

}

extern "C" void thrust_sort_keys_int(float* inputs, int* keys, unsigned int nbElements,  unsigned int offset)
{

    const thrust::device_ptr<float> thrust_data(inputs);
    const thrust::device_ptr<int> thrust_keys(keys);

    thrust::stable_sort_by_key( thrust_data + offset,
                                thrust_data + offset + nbElements,
                                thrust_keys + offset,
                                thrust::greater<float>());

}

extern "C" void thrust_gather(const unsigned int* keys,
                         const float* inputs,
                         float* outputs,
                         unsigned int nbElements,
                         unsigned int inputOffset,
                         unsigned int outputOffset)
{
    const thrust::device_ptr<const float> thrust_data_inputs(inputs);
    const thrust::device_ptr<float> thrust_data_outputs(outputs);
    const thrust::device_ptr<const unsigned int> thrust_keys(keys);

    thrust::gather( thrust_keys + inputOffset,
                    thrust_keys + inputOffset + nbElements,
                    thrust_data_inputs + inputOffset,
                    thrust_data_outputs + outputOffset);

}

extern "C" void thrust_gather_int(const int* keys,
                         const float* inputs,
                         float* outputs,
                         unsigned int nbElements,
                         unsigned int inputOffset,
                         unsigned int outputOffset)
{
    const thrust::device_ptr<const float> thrust_data_inputs(inputs);
    const thrust::device_ptr<float> thrust_data_outputs(outputs);
    const thrust::device_ptr<const int> thrust_keys(keys);

    thrust::gather( thrust_keys + inputOffset,
                    thrust_keys + inputOffset + nbElements,
                    thrust_data_inputs + inputOffset,
                    thrust_data_outputs + outputOffset);

}
extern "C" void thrust_gather_int2int(  const int* keys,
                                        const int* inputs,
                                        int* outputs,
                                        unsigned int nbElements,
                                        unsigned int inputOffset,
                                        unsigned int outputOffset)
{
    const thrust::device_ptr<const int> thrust_data_inputs(inputs);
    const thrust::device_ptr<int> thrust_data_outputs(outputs);
    const thrust::device_ptr<const int> thrust_keys(keys);

    thrust::gather( thrust_keys + inputOffset,
                    thrust_keys + inputOffset + nbElements,
                    thrust_data_inputs + inputOffset,
                    thrust_data_outputs + outputOffset);

}

extern "C" void thrust_scatter_int2int(  const int* keys,
                                        const int* inputs,
                                        int* outputs,
                                        unsigned int nbElements)
{
    const thrust::device_ptr<const int> thrust_data_inputs(inputs);
    const thrust::device_ptr<int> thrust_data_outputs(outputs);
    const thrust::device_ptr<const int> thrust_keys(keys);

    thrust::scatter(    thrust_keys,
                        thrust_keys + nbElements,
                        thrust_data_inputs,
                        thrust_data_outputs);

}