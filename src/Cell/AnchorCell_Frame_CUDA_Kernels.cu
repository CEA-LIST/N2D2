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

#include "Cell/AnchorCell_Frame_CUDA_Kernels.hpp"

// atomicMax() does not exist in CUDA for floats
// Source: https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* addressAsInt = (int*) address;
    int old = *addressAsInt;
    int assumed;

    do {
        assumed = old;
        old = ::atomicCAS(addressAsInt, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    }
    while (assumed != old);

    return __int_as_float(old);
}

__global__ void cudaSAnchorPropagate_kernel(
    unsigned int stimuliSizeX,
    unsigned int stimuliSizeY,
    bool flip,
    bool inference,
    float* inputsCls,
    float* inputsCoord,
    unsigned int scoresCls,
    N2D2::AnchorCell_Frame_Kernels::Anchor* anchors,
    N2D2::AnchorCell_Frame_Kernels::BBox_T** gts,
    unsigned int* nbLabels,
    float* outputs,
    int* argMaxIoU,
    float* maxIoU,
    unsigned int nbAnchors,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    unsigned int nbTotalCls,
    unsigned int nbInputs)
{
    const unsigned int batchOffset = blockIdx.z * 6 * nbAnchors
                                        * outputsHeight * outputsWidth;
    
    const unsigned int batchCoordOffset = blockIdx.z * 4 * nbAnchors 
                                        * outputsHeight * outputsWidth;

    const unsigned int batchClsOffset = blockIdx.z * nbAnchors * nbTotalCls
                                        * outputsHeight * outputsWidth;


    const float xRatio = ceil(stimuliSizeX / (float)outputsWidth);
    const float yRatio = ceil(stimuliSizeY / (float)outputsHeight);
    float globalMaxIoU = 0.0;
    
    for (unsigned int k = blockIdx.x; k < nbAnchors;
         k += gridDim.x)
    {
        for (unsigned int ya = threadIdx.y; ya < outputsHeight;
             ya += blockDim.y)
        {
            for (unsigned int xa = threadIdx.x; xa < outputsWidth;
                 xa += blockDim.x)
            {
                
                // Shifted anchors coordinates at (xa, ya)
                const int xa0 = (int)(anchors[k].x0 + xa * xRatio);
                const int ya0 = (int)(anchors[k].y0 + ya * yRatio);
                const int xa1 = (int)(anchors[k].x1 + xa * xRatio);
                const int ya1 = (int)(anchors[k].y1 + ya * yRatio);
/*
                const int xa0 = (int)(anchors[k*4 + 0] + xa * xRatio);
                const int ya0 = (int)(anchors[k*4 + 1] + ya * yRatio);
                const int xa1 = (int)(anchors[k*4 + 2] + xa * xRatio);
                const int ya1 = (int)(anchors[k*4 + 3] + ya * yRatio);
*/
                const int wa = xa1 - xa0;
                const int ha = ya1 - ya0;

                // Anchor center coordinates (xac, yac)
                const float xac = xa0 + wa / 2.0;
                const float yac = ya0 + ha / 2.0;

                const unsigned int addrBase = batchOffset
                    + xa + (ya + k * outputsHeight) * outputsWidth;

                const unsigned int addrCoordBase = batchCoordOffset + k * outputsWidth * outputsHeight + ya * outputsWidth + xa;

                const unsigned int addrClsBase = batchClsOffset + k * outputsWidth * outputsHeight + ya * outputsWidth + xa;

                const unsigned int addrStep = outputsHeight * outputsWidth;


                /**
                 * 1st condition: "During  training,  we  ignore all
                 * cross-boundary anchors so they do not contribute to  the
                 * loss."
                 * 2nd condition: "During testing, however, we still apply
                 * the fully convolutional RPN  to  the  entire  image."
                */

                if ((xa0 >= 0 && ya0 >= 0 && xa1 < (int)stimuliSizeX && ya1 < (int)stimuliSizeY) || inference)
                {

                    // Score
                    //const float cls = inputsCls[addrBase];
                    // Parameterized coordinates
                    //const float txbb = inputsCoord[addrBase + scoresCls * nbAnchors * addrStep];
                    //const float tybb = inputsCoord[addrBase + (scoresCls + 1) * nbAnchors * addrStep];
                    //const float twbb = inputsCoord[addrBase + (scoresCls + 2) * nbAnchors * addrStep];
                    //const float thbb = inputsCoord[addrBase + (scoresCls + 3) * nbAnchors * addrStep];

                        

                    // Score
                    const float cls = inputsCls[addrClsBase];
                    // Parameterized coordinates
                   
                    const float txbb = inputsCoord[addrCoordBase + scoresCls * nbAnchors * addrStep];
                    const float tybb = inputsCoord[addrCoordBase + (scoresCls + 1) * nbAnchors * addrStep];
                    const float twbb = inputsCoord[addrCoordBase + (scoresCls + 2) * nbAnchors * addrStep];
                    const float thbb = inputsCoord[addrCoordBase + (scoresCls + 3) * nbAnchors * addrStep];

                    // Predicted box center coordinates
                    const float xbbc = ((flip) ? -txbb : txbb) * wa
                                            + xac;
                    const float ybbc = ((flip) ? -tybb : tybb) * ha
                                            + yac;
                    float wbb = wa * exp(twbb);
                    float hbb = ha * exp(thbb);

                    // Predicted box top-left coordinates
                    float xbb = xbbc - wbb / 2.0;
                    float ybb = ybbc - hbb / 2.0;

                    if (inference) {
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
                        if (xbb + wbb > stimuliSizeX - 1)
                            wbb = stimuliSizeX - 1 - xbb;
                        if (ybb + hbb > stimuliSizeY - 1)
                            hbb = stimuliSizeY - 1 - ybb;
                    }

                    // For inference, compute IoU on predicted boxes
                    // For learning, compute IoU on anchor boxes
                    // => if IoU is computed on predicted boxes during
                    // learning, predicted boxes may arbitrarily drift from
                    // anchors and learning does not converge

                    const N2D2::AnchorCell_Frame_Kernels::BBox_T bb
                        = (inference)
                            ? N2D2::AnchorCell_Frame_Kernels::BBox_T
                                (xbb, ybb, wbb, hbb)
                            : N2D2::AnchorCell_Frame_Kernels::BBox_T
                                (xa0, ya0, wa, ha);
                    
                    float maxIoU_ = 0.0;
                    int argMaxIoU_ = -1;


                    for (unsigned int l = 0; l < nbLabels[blockIdx.z]; ++l) {
                        // Ground Truth box coordinates
                        const N2D2::AnchorCell_Frame_Kernels::BBox_T& gt
                            = gts[blockIdx.z][l];

                        const float interLeft = max(gt.x, bb.x);
                        const float interRight = min(gt.x + gt.w, bb.x + bb.w);
                        const float interTop = max(gt.y, bb.y);
                        const float interBottom = min(gt.y + gt.h, bb.y + bb.h);

                        if (interLeft < interRight
                            && interTop < interBottom)
                        {
                            const float interArea
                                = (interRight - interLeft)
                                    * (interBottom - interTop);
                            const float unionArea = gt.w * gt.h
                                + bb.w * bb.h - interArea;
                            const float IoU = interArea / unionArea;

                            if (IoU > maxIoU_) {
                                maxIoU_ = IoU;
                                argMaxIoU_ = l;
                            }
                        }
                        
                    }

                    outputs[addrBase] = cls;
                    outputs[addrBase + 1 * nbAnchors * addrStep] = xbb;
                    outputs[addrBase + 2 * nbAnchors * addrStep] = ybb;
                    outputs[addrBase + 3 * nbAnchors * addrStep] = wbb;
                    outputs[addrBase + 4 * nbAnchors * addrStep] = hbb;
                    outputs[addrBase + 5 * nbAnchors * addrStep] = maxIoU_;

                    argMaxIoU[addrBase] = argMaxIoU_;
                    globalMaxIoU = max(globalMaxIoU, maxIoU_);
                    
                }
                else {
                    
                    outputs[addrBase] = -1.0;
                    outputs[addrBase + 1 * nbAnchors * addrStep] = 0.0;
                    outputs[addrBase + 2 * nbAnchors * addrStep] = 0.0;
                    outputs[addrBase + 3 * nbAnchors * addrStep] = 0.0;
                    outputs[addrBase + 4 * nbAnchors * addrStep] = 0.0;
                    outputs[addrBase + 5 * nbAnchors * addrStep] = 0.0;
                    argMaxIoU[addrBase] = -1;
                    
                }
            }
        }
    }

    atomicMax(maxIoU + blockIdx.z, globalMaxIoU);
}

static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}

void N2D2::cudaSAnchorPropagate(
    unsigned int stimuliSizeX,
    unsigned int stimuliSizeY,
    bool flip,
    bool inference,
    float* inputsCls,
    float* inputsCoord,
    unsigned int scoresCls,
    AnchorCell_Frame_Kernels::Anchor* anchors,
    AnchorCell_Frame_Kernels::BBox_T** gts,
    unsigned int* nbLabels,
    float* outputs,
    int* argMaxIoU,
    float* maxIoU,
    unsigned int nbAnchors,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    unsigned int nbTotalCls,
    unsigned int nbInputs)
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

    const dim3 blocksPerGrid = {nbAnchors, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSAnchorPropagate_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (stimuliSizeX,
           stimuliSizeY,
           flip,
           inference,
           inputsCls,
           inputsCoord,
           scoresCls,
           anchors,
           gts,
           nbLabels,
           outputs,
           argMaxIoU,
           maxIoU,
           nbAnchors,
           outputsHeight,
           outputsWidth,
           batchSize,
           nbTotalCls,
           nbInputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}
