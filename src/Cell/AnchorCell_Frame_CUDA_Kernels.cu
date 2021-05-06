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
    unsigned int featureMapX,
    unsigned int featureMapY,
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


    const float xRatio = ceil(featureMapX / (float)outputsWidth);
    const float yRatio = ceil(featureMapY / (float)outputsHeight);
    const float xOutputRatio = stimuliSizeX / (float) (featureMapX - 1.0);
    const float yOutputRatio = stimuliSizeY / (float) (featureMapY - 1.0);

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

                const int wa = xa1 - xa0;
                const int ha = ya1 - ya0;

                // Anchor center coordinates (xac, yac)
                const float xac = xa0 + wa / 2.0;
                const float yac = ya0 + ha / 2.0;

                const unsigned int addrStep = outputsHeight * outputsWidth;

                const unsigned int addrBase = batchOffset + xa + (ya + k * outputsHeight) * outputsWidth;

                const unsigned int addrCoordBase = batchCoordOffset 
                                                    + k * addrStep
                                                    + ya * outputsWidth 
                                                    + xa;

                const unsigned int addrClsBase = batchClsOffset 
                                                    + k * addrStep 
                                                    + ya * outputsWidth 
                                                    + xa;



                /**
                 * 1st condition: "During  training,  we  ignore all
                 * cross-boundary anchors so they do not contribute to  the
                 * loss."
                 * 2nd condition: "During testing, however, we still apply
                 * the fully convolutional RPN  to  the  entire  image."
                */

                if ((xa0 >= 0 && ya0 >= 0 && xa1 < (int)featureMapX && ya1 < (int)featureMapY) || inference)
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
                        if (xbb + wbb > featureMapX - 1)
                            wbb = featureMapX - 1 - xbb;
                        if (ybb + hbb > featureMapY - 1)
                            hbb = featureMapY - 1 - ybb;
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

                    /*
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
                    */
                    //Rescale Bounding Box if Feature MAP size is different than stimuli size
                    xbb *=  xOutputRatio;
                    wbb *=  xOutputRatio;
                    ybb *=  yOutputRatio;
                    hbb *=  yOutputRatio;
                    
                    outputs[addrBase] = cls;
                    outputs[addrBase + 1 * nbAnchors * addrStep] = xbb;
                    outputs[addrBase + 2 * nbAnchors * addrStep] = ybb;
                    outputs[addrBase + 3 * nbAnchors * addrStep] = wbb;
                    outputs[addrBase + 4 * nbAnchors * addrStep] = hbb;
                    outputs[addrBase + 5 * nbAnchors * addrStep] = maxIoU_;

                    argMaxIoU[batchClsOffset/nbTotalCls + k * addrStep + ya * outputsWidth + xa] = argMaxIoU_;
                                
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



void N2D2::cudaSAnchorPropagate(
    unsigned int stimuliSizeX,
    unsigned int stimuliSizeY,
    unsigned int featureMapX,
    unsigned int featureMapY,
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
    unsigned int nbInputs,
    const dim3 blocksPerGrid,
    const dim3 threadsPerBlock)
{
    cudaSAnchorPropagate_kernel<<<blocksPerGrid, threadsPerBlock>>>
        (stimuliSizeX,
           stimuliSizeY,
           featureMapX,
           featureMapY,
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

__global__ void cudaSAnchorPropagate_LapNet_kernel( unsigned int stimuliSizeX,
                                                    unsigned int stimuliSizeY,
                                                    unsigned int featureMapX,
                                                    unsigned int featureMapY,
                                                    bool flip,
                                                    bool inference,
                                                    float* inputsCls,
                                                    float* inputsCoord,
                                                    bool isCoordinateAnchor,
                                                    unsigned int scoresCls,
                                                    N2D2::AnchorCell_Frame_Kernels::Anchor* anchors,
                                                    N2D2::AnchorCell_Frame_Kernels::BBox_T* gtsWithClass,
                                                    unsigned int* nbLabelsWithClass,
                                                    float* outputs,
                                                    int* argMaxIoU,
                                                    float* maxIoU,
                                                    unsigned int nbAnchors,
                                                    unsigned int outputsHeight,
                                                    unsigned int outputsWidth,
                                                    unsigned int batchSize,
                                                    unsigned int nbTotalCls,
                                                    unsigned int nbClass,
                                                    unsigned int nbInputs,
                                                    unsigned int maxGTLabels)
{
    const unsigned int batchOffset = blockIdx.z * 6 * nbAnchors
                                        * outputsHeight * outputsWidth;
    
    const unsigned int batchCoordOffset = blockIdx.z * 4 * nbAnchors 
                                        * outputsHeight * outputsWidth;

    const unsigned int batchClsOffset = blockIdx.z * nbAnchors
                                        * outputsHeight * outputsWidth;


    const float xRatio = ceil(featureMapX / (float)outputsWidth);
    const float yRatio = ceil(featureMapY / (float)outputsHeight);
    /*const float xOutputRatio = stimuliSizeX / (float) featureMapX;
    const float yOutputRatio = stimuliSizeY / (float) featureMapY;
   */
    const float xOutputRatio = stimuliSizeX;
    const float yOutputRatio = stimuliSizeY;

    for (unsigned int k = blockIdx.x; k < nbAnchors;
         k += gridDim.x)
    {
        const int classIdx = k/(nbAnchors/nbClass);

        for (unsigned int ya = threadIdx.y; ya < outputsHeight;
             ya += blockDim.y)
        {
            for (unsigned int xa = threadIdx.x; xa < outputsWidth;
                 xa += blockDim.x)
            {
                
                // Shifted anchors coordinates at (xa, ya)
                /*const int xa0 = (int)(anchors[k].x0 + xa * xRatio);
                const int ya0 = (int)(anchors[k].y0 + ya * yRatio);
                const int xa1 = (int)(anchors[k].x1 + xa * xRatio);
                const int ya1 = (int)(anchors[k].y1 + ya * yRatio);

                const int wa = xa1 - xa0;
                const int ha = ya1 - ya0;*/
                const float xa0 = (anchors[k].x0 + xa * xRatio)
                                    / (float)(featureMapX - 1.0);
                const float ya0 = (anchors[k].y0 + ya * yRatio)
                                     / (float) (featureMapY - 1.0);
                const float xa1 = (anchors[k].x1 + xa * xRatio)
                                    / (float)(featureMapX - 1.0);
                const float ya1 = (anchors[k].y1 + ya * yRatio)
                                    / (float) (featureMapY - 1.0);

                const float wa = xa1 - xa0;
                const float ha = ya1 - ya0;
                // Anchor center coordinates (xac, yac)
                const float xac = xa0 + wa / 2.0;
                const float yac = ya0 + ha / 2.0;

                const unsigned int addrStep = outputsHeight * outputsWidth;

                const unsigned int addrBase = batchOffset + xa + (ya + k * outputsHeight) * outputsWidth;

                const unsigned int addrCoordBase = isCoordinateAnchor ? batchCoordOffset 
                                                                        + k * addrStep
                                                                        + ya * outputsWidth 
                                                                        + xa :
                                                                        batchCoordOffset 
                                                                        + k * addrStep * 4
                                                                        + ya * outputsWidth 
                                                                        + xa ;


                const unsigned int addrClsBase = batchClsOffset 
                                                + k * addrStep 
                                                + ya * outputsWidth 
                                                + xa;
                const unsigned int index_xbb = isCoordinateAnchor ? 
                                              addrCoordBase + scoresCls * nbAnchors * addrStep 
                                              : addrCoordBase + 0 * addrStep;
                const unsigned int index_ybb = isCoordinateAnchor ? 
                                              addrCoordBase + (scoresCls + 1) * nbAnchors * addrStep 
                                              : addrCoordBase + 1 * addrStep;
                const unsigned int index_wbb = isCoordinateAnchor ? 
                                              addrCoordBase + (scoresCls + 2) * nbAnchors * addrStep
                                              : addrCoordBase + 2 * addrStep;
                const unsigned int index_hbb =  isCoordinateAnchor ? 
                                                addrCoordBase + (scoresCls + 3) * nbAnchors * addrStep 
                                                : addrCoordBase + 3 * addrStep;

                // Score
                const float cls = inputsCls[addrClsBase];
                // Parameterized coordinates
                const float txbb = fmaxf(fminf(inputsCoord[index_xbb], 70.0f), -70.0f);
                const float tybb = fmaxf(fminf(inputsCoord[index_ybb], 70.0f), -70.0f);
                const float twbb = fmaxf(fminf(inputsCoord[index_wbb], 70.0f), -70.0f);
                const float thbb = fmaxf(fminf(inputsCoord[index_hbb], 70.0f), -70.0f);

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

                xbb *= xOutputRatio;
                wbb *= xOutputRatio;
                ybb *= yOutputRatio;
                hbb *= yOutputRatio;

                // For inference, compute IoU on predicted boxes
                // For learning, compute IoU on anchor boxes
                // => if IoU is computed on predicted boxes during
                // learning, predicted boxes may arbitrarily drift from
                // anchors and learning does not converge

                const N2D2::AnchorCell_Frame_Kernels::BBox_T bb = (inference)
                    ? N2D2::AnchorCell_Frame_Kernels::BBox_T(xbb, ybb, wbb, hbb)
                    : N2D2::AnchorCell_Frame_Kernels::BBox_T(
                          xa0 * xOutputRatio, ya0 * yOutputRatio, wa * xOutputRatio, ha * yOutputRatio);

                // Renormalize the bounding boxes


                float maxIoU_ = 0.0;
                int argMaxIoU_ = -1;
                
                for (unsigned int l = 0; l < nbLabelsWithClass[blockIdx.z*nbClass + classIdx]; ++l) {
                    // Ground Truth box coordinates
                    const N2D2::AnchorCell_Frame_Kernels::BBox_T gt 
                        = gtsWithClass[blockIdx.z*maxGTLabels*nbClass +  classIdx*maxGTLabels + l];
                    
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

                argMaxIoU[addrClsBase] = argMaxIoU_;
            }
        }
    }
}


void N2D2::cudaSAnchorPropagate_LapNet( unsigned int stimuliSizeX,
                                        unsigned int stimuliSizeY,
                                        unsigned int featureMapX,
                                        unsigned int featureMapY,
                                        bool flip,
                                        bool inference,
                                        float* inputsCls,
                                        float* inputsCoord,
                                        bool isCoordinateAnchor,
                                        unsigned int scoresCls,
                                        AnchorCell_Frame_Kernels::Anchor* anchors,
                                        AnchorCell_Frame_Kernels::BBox_T* gtsWithClass,
                                        unsigned int* nbLabelsWithClass,
                                        float* outputs,
                                        int* argMaxIoU,
                                        float* maxIoU,
                                        unsigned int nbAnchors,
                                        unsigned int outputsHeight,
                                        unsigned int outputsWidth,
                                        unsigned int batchSize,
                                        unsigned int nbTotalCls,
                                        unsigned int nbClass,
                                        unsigned int nbInputs,
                                        unsigned int maxLabelGT,
                                        const dim3 blocksPerGrid,
                                        const dim3 threadsPerBlock)
{
    cudaSAnchorPropagate_LapNet_kernel<<<blocksPerGrid, threadsPerBlock>>> (stimuliSizeX,
                                                                            stimuliSizeY,
                                                                            featureMapX,
                                                                            featureMapY,
                                                                            flip,
                                                                            inference,
                                                                            inputsCls,
                                                                            inputsCoord,
                                                                            isCoordinateAnchor,
                                                                            scoresCls,
                                                                            anchors,
                                                                            gtsWithClass,
                                                                            nbLabelsWithClass,
                                                                            outputs,
                                                                            argMaxIoU,
                                                                            maxIoU,
                                                                            nbAnchors,
                                                                            outputsHeight,
                                                                            outputsWidth,
                                                                            batchSize,
                                                                            nbTotalCls,
                                                                            nbClass,
                                                                            nbInputs,
                                                                            maxLabelGT);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

__global__ void cudaSAnchorBackPropagateSSD_kernel(  
                                                    const float* inputsCls,
                                                    const float* outputs,
                                                    const int* argMaxIoU,
                                                    float* diffOutputsCoords,
                                                    float* diffOutputsCls,
                                                    int* keyNegSamples,
                                                    int* keyPosSamples,
                                                    float* confNegSamples,
                                                    float* confPosSamples,
                                                    float positiveIoU,
                                                    unsigned int nbAnchors,
                                                    unsigned int outputsHeight,
                                                    unsigned int outputsWidth,
                                                    unsigned int batchSize,
                                                    unsigned int nbTotalCls,
                                                    unsigned int nbClass,
                                                    unsigned int nbInputs)
{
    const unsigned int batchOffset = blockIdx.z * 6 * nbAnchors * outputsHeight * outputsWidth;
    const unsigned int batchClsOffset = blockIdx.z * nbAnchors * outputsHeight * outputsWidth;
    const unsigned int batchCoordOffset = blockIdx.z * 4 * nbAnchors  * outputsHeight * outputsWidth;

    for (unsigned int k = blockIdx.x; k < nbAnchors; k += gridDim.x)
    {
        const int classIdx = (k/nbAnchors)*nbClass;
        const int classOffset = outputsHeight*outputsWidth*nbAnchors*classIdx;

        for (unsigned int ya = threadIdx.y; ya < outputsHeight; ya += blockDim.y)
        {
            for (unsigned int xa = threadIdx.x; xa < outputsWidth; xa += blockDim.x)
            {
                const unsigned int addrStep = outputsHeight * outputsWidth;

                const unsigned int addrBase = batchOffset + xa + (ya + k * outputsHeight) * outputsWidth;

                const unsigned int addrClsBase = batchClsOffset 
                                                    + k * addrStep 
                                                    + ya * outputsWidth 
                                                    + xa;
                const unsigned int addrCoordBase = batchCoordOffset 
                                                    + k * addrStep
                                                    + ya * outputsWidth 
                                                    + xa;

                // Score
                const float conf = inputsCls[addrClsBase];
                const float IoU = outputs[addrBase + 5 * nbAnchors * addrStep];

                const int maxIoU = argMaxIoU[addrClsBase];

                if (IoU >= positiveIoU && maxIoU > -1)
                {
                    keyPosSamples[addrClsBase] = xa + (ya + k * outputsHeight) * outputsWidth;
                    confPosSamples[addrClsBase] = conf;

                    keyNegSamples[addrClsBase] = -1;
                    confNegSamples[addrClsBase] = -1.0f;

                }
                else if(maxIoU == -1 /*&& IoU <= 0.0*/)
                {
                    keyNegSamples[addrClsBase] = xa + (ya + k * outputsHeight) * outputsWidth;
                    confNegSamples[addrClsBase] = conf;

                    keyPosSamples[addrClsBase] = -1;
                    confPosSamples[addrClsBase] = -1.0f;
                }
                else
                {
                    //printf("maxIoU: %d\n", maxIoU);

                    keyNegSamples[addrClsBase] = -1;
                    confNegSamples[addrClsBase] = -1.0f;

                    keyPosSamples[addrClsBase] = -1;
                    confPosSamples[addrClsBase] = -1.0f;

                }
                


                diffOutputsCls[addrClsBase] = 0.0f;

                diffOutputsCoords[addrCoordBase] = 0.0f;
                diffOutputsCoords[addrCoordBase + 1 * nbAnchors * addrStep] = 0.0f;
                diffOutputsCoords[addrCoordBase + 2 * nbAnchors * addrStep] = 0.0f;
                diffOutputsCoords[addrCoordBase + 3 * nbAnchors * addrStep] = 0.0f;

            }
        }
    }
}

void N2D2::cudaSAnchorBackPropagateSSD(const float* inputsCls,
                                                const float* outputs,
                                                const int* argMaxIoU,
                                                float* diffOutputsCoords,
                                                float* diffOutputsCls,
                                                int* keyNegSamples,
                                                int* keyPosSamples,
                                                float* confNegSamples,
                                                float* confPosSamples,
                                                float positiveIoU,
                                                unsigned int nbAnchors,
                                                unsigned int outputsHeight,
                                                unsigned int outputsWidth,
                                                unsigned int batchSize,
                                                unsigned int nbTotalCls,
                                                unsigned int nbClass,
                                                unsigned int nbInputs,
                                                const dim3 blocksPerGrid,
                                                const dim3 threadsPerBlock)
{
    cudaSAnchorBackPropagateSSD_kernel<<<blocksPerGrid, threadsPerBlock>>> (inputsCls,
                                                                            outputs,
                                                                            argMaxIoU,
                                                                            diffOutputsCoords,
                                                                            diffOutputsCls,
                                                                            keyNegSamples,
                                                                            keyPosSamples,
                                                                            confNegSamples,
                                                                            confPosSamples,
                                                                            positiveIoU,
                                                                            nbAnchors,
                                                                            outputsHeight,
                                                                            outputsWidth,
                                                                            batchSize,
                                                                            nbTotalCls,
                                                                            nbClass,
                                                                            nbInputs);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


__global__ void cudaSAnchorBackPropagateSSD_NegSamples_kernel(const float* inputCls,
                                                                float* diffOutputsCls,
                                                                const float* confSamples,
                                                                const int* keySamples,
                                                                const int nbSamples,
                                                                const int nbPositive,
                                                                const unsigned int nbAnchors,
                                                                const unsigned int outputsHeight,
                                                                const unsigned int outputsWidth,
                                                                const unsigned int batchSize)
{

    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

    if(index < nbSamples)
    {
        const int indexSamples = keySamples[index];
        const float error = inputCls[indexSamples];
        diffOutputsCls[indexSamples] = -error / (nbPositive);

    }
    
}


void N2D2::cudaSAnchorBackPropagate_SSD_NegSamples( const float* inputCls,
                                                    float* diffOutputsCls,
                                                    const float* confNegSamples,
                                                    const int* keyNegSamples,
                                                    const int nbNegative,
                                                    const int nbPositive,
                                                    const unsigned int nbAnchors,
                                                    const unsigned int outputsHeight,
                                                    const unsigned int outputsWidth,
                                                    const unsigned int batchSize,
                                                    const dim3 blocksPerGrid,
                                                    const dim3 threadsPerBlock)
{

    cudaSAnchorBackPropagateSSD_NegSamples_kernel<<<blocksPerGrid, threadsPerBlock>>> (inputCls,
                                                                                        diffOutputsCls,
                                                                                        confNegSamples,
                                                                                        keyNegSamples,
                                                                                        nbNegative,
                                                                                        nbPositive,
                                                                                        nbAnchors,
                                                                                        outputsHeight,
                                                                                        outputsWidth,
                                                                                        batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}



__device__ static float smoothL1(float tx, float x)
{
    const float error = tx - x;
    float return_value = 0.0f;
    float sign = -1.0f;
    if (error >= 0.0f)
        sign = 1.0f;
    
    /*
float sign = 1.0f;


    if (error >= 0.0f)
        sign = 1.0f;

    if(abs(error) >= 1.0f)
        return_value = sign;
        */
    if(abs(error) < 1.0f)
    {
        return_value = sign * 0.5*(abs(error)*abs(error));
    }
    else
        return_value = (abs(error) - 0.5f)*sign;
    return return_value;
}

__device__ static float huberLoss(float tx, float x, float sigma= 1.0f)
{
    float return_value = 0.0f;

    const float diff = tx - x;
    const float sigma2 = sigma*sigma;
    const float posCas = (diff*diff) * 0.5*sigma2;
    const float negCas = abs(diff) - (0.5 / sigma2) ;

    if(abs(diff) < (1.0/sigma2)) {
        return_value = posCas;
    } 
    else {
        return_value = negCas;
    }

    return return_value;
}

__global__ void cudaSAnchorBackPropagateSSD_PosSamples_kernel(  const unsigned int stimuliSizeX,
                                                                const unsigned int stimuliSizeY,
                                                                const unsigned int featureMapX,
                                                                const unsigned int featureMapY,
                                                                const float* inputCls,
                                                                float* diffOutputsCls,
                                                                const float* inputCoord,
                                                                float* diffOutputsCoord,
                                                                const float* confSamples,
                                                                const int* keySamples,
                                                                N2D2::AnchorCell_Frame_Kernels::Anchor* anchors,
                                                                const float xRatio,
                                                                const float yRatio,
                                                                N2D2::AnchorCell_Frame_Kernels::BBox_T* gtsWithClass,
                                                                const int* argMaxIoU,
                                                                const int nbPositive,
                                                                const float lossLambda,
                                                                const unsigned int nbAnchors,
                                                                const unsigned int outputsHeight,
                                                                const unsigned int outputsWidth,
                                                                const unsigned int batchSize)
{

    const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;
    /*const float xOutputRatio = stimuliSizeX / (float) featureMapX;
    const float yOutputRatio = stimuliSizeY / (float) featureMapY;
   */
    const float xOutputRatio = stimuliSizeX;
    const float yOutputRatio = stimuliSizeY;

    if(index < nbPositive)
    {
        const int indexSamples = keySamples[index];
        const int keyArgMax = argMaxIoU[indexSamples];
        N2D2::AnchorCell_Frame_Kernels::BBox_T gt = gtsWithClass[keyArgMax];
        gt.x /= xOutputRatio;
        gt.w /= xOutputRatio;
        gt.y /= yOutputRatio;
        gt.h /= yOutputRatio;

        const int xa = indexSamples % outputsWidth;
        const int k = indexSamples / (outputsHeight * outputsWidth);
        const int ya = (indexSamples - k*outputsHeight*outputsWidth) / outputsWidth;

        /*const int xa0 = (int)(anchors[k].x0 + xa * xRatio);
        const int ya0 = (int)(anchors[k].y0 + ya * yRatio);
        const int xa1 = (int)(anchors[k].x1 + xa * xRatio);
        const int ya1 = (int)(anchors[k].y1 + ya * yRatio);

        // Anchors width and height
        const int wa = xa1 - xa0;
        const int ha = ya1 - ya0;*/
        const float xa0
            = (anchors[k].x0 + xa * xRatio) / (float)(featureMapX - 1.0);
        const float ya0 
            = (anchors[k].y0 + ya * yRatio) / (float)(featureMapY - 1.0);
        const float xa1
            = (anchors[k].x1 + xa * xRatio) / (float)(featureMapX - 1.0);
        
        const float ya1 
            = (anchors[k].y1 + ya * yRatio) / (float)(featureMapY - 1.0);

        // Anchors width and height
        const float wa = xa1 - xa0;
        const float ha = ya1 - ya0;

        // Anchor center coordinates (xac, yac)
        const float xac = xa0 + wa / 2.0;
        const float yac = ya0 + ha / 2.0;

        // Ground Truth center coordinates (xgtc, ygtc) and normalization between 0.0 and 1.0
        const float xgtc = (gt.x + gt.w / 2.0) ;
        const float ygtc = (gt.y + gt.h / 2.0) ;

        // Parameterized Ground Truth center coordinates
        const float txgt = (xgtc - xac) / wa;
        const float tygt =(ygtc - yac) / ha;
        const float twgt = log(gt.w / wa);
        const float thgt = log(gt.h / ha);

        // Parameterized coordinates
        const float tx = inputCoord[indexSamples];
        const float ty = inputCoord[indexSamples + 1 * outputsHeight * outputsWidth * nbAnchors];
        const float tw = inputCoord[indexSamples + 2 * outputsHeight * outputsWidth * nbAnchors];
        const float th = inputCoord[indexSamples + 3 * outputsHeight * outputsWidth * nbAnchors];



        // Smooth L1 loss
        const float lossTx = lossLambda * huberLoss(txgt, tx) / (nbPositive); 
        const float lossTy = lossLambda * huberLoss(tygt, ty) / (nbPositive); 
        const float lossTw = lossLambda * huberLoss(twgt, tw) / (nbPositive); 
        const float lossTh = lossLambda * huberLoss(thgt, th) / (nbPositive); 
        const float lossCls = (1.0f - inputCls[indexSamples]) / (nbPositive); 
        //printf("Coord[%d]: {%f, %f, %f, %f}(%f)\n", indexSamples, lossTx,
        //    lossTy, lossTw, lossTh, inputCls[indexSamples]);

        diffOutputsCoord[indexSamples] = lossTx;
        diffOutputsCoord[indexSamples + 1 * outputsHeight * outputsWidth * nbAnchors] = lossTy;
        diffOutputsCoord[indexSamples + 2 * outputsHeight * outputsWidth * nbAnchors] = lossTw;
        diffOutputsCoord[indexSamples + 3 * outputsHeight * outputsWidth * nbAnchors] = lossTh;
        diffOutputsCls[indexSamples] = lossCls; 
    }
    
}

void N2D2::cudaSAnchorBackPropagateSSD_PosSamples(  const unsigned int stimuliSizeX,
                                                    const unsigned int stimuliSizeY,
                                                    const unsigned int featureMapX,
                                                    const unsigned int featureMapY,
                                                    const float* inputCls,
                                                    float* diffOutputsCls,
                                                    const float* inputCoord,
                                                    float* diffOutputsCoord,
                                                    const float* confSamples,
                                                    const int* keySamples,
                                                    N2D2::AnchorCell_Frame_Kernels::Anchor* anchors,
                                                    const float xRatio,
                                                    const float yRatio,
                                                    N2D2::AnchorCell_Frame_Kernels::BBox_T* gtsWithClass,
                                                    const int* argMaxIoU,
                                                    const int nbPositive,
                                                    const float lossLambda,
                                                    const unsigned int nbAnchors,
                                                    const unsigned int outputsHeight,
                                                    const unsigned int outputsWidth,
                                                    const unsigned int batchSize,
                                                    const dim3 blocksPerGrid,
                                                    const dim3 threadsPerBlock)
{

    cudaSAnchorBackPropagateSSD_PosSamples_kernel<<<blocksPerGrid, threadsPerBlock>>> (stimuliSizeX,
                                                                                        stimuliSizeY,
                                                                                        featureMapX,
                                                                                        featureMapY,
                                                                                        inputCls,
                                                                                        diffOutputsCls,
                                                                                        inputCoord,
                                                                                        diffOutputsCoord,
                                                                                        confSamples,
                                                                                        keySamples,
                                                                                        anchors,
                                                                                        xRatio,
                                                                                        yRatio,
                                                                                        gtsWithClass,
                                                                                        argMaxIoU,
                                                                                        nbPositive,
                                                                                        lossLambda,
                                                                                        nbAnchors,
                                                                                        outputsHeight,
                                                                                        outputsWidth,
                                                                                        batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}