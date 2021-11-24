/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#include "kernels_cpu.hpp"

void anchor_cpu(unsigned int batchSize,
                unsigned int nbOutputs,
                unsigned int outputHeight,
                unsigned int outputWidth,
                unsigned int stimuliHeight,
                unsigned int stimuliWidth,
                unsigned int scoreCls,
                bool isFlip,
                unsigned int nbAnchors,
                double xRatio,
                double yRatio,
                std::vector<Anchor> anchors,
                const float* inputs,
                float* outputs/*,
                float* maxIoU,
                float* ArgMaxIoU*/)
{

    const unsigned int size = batchSize * nbAnchors;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (batchSize > 4 && size > 16)
#endif

    for (int batchPos = 0; batchPos < (int)batchSize; ++batchPos) {
        for (unsigned int k = 0; k < nbAnchors; ++k) {
            const Anchor& anchor = anchors[k];

            for (unsigned int ya = 0; ya < outputHeight; ++ya) {
                for (unsigned int xa = 0; xa < outputWidth; ++xa) {

                    // Shifted anchors coordinates at (xa, ya)
                    const int xa0 = (int)(anchor.x0 + xa * xRatio);
                    const int ya0 = (int)(anchor.y0 + ya * yRatio);
                    const int xa1 = (int)(anchor.x1 + xa * xRatio);
                    const int ya1 = (int)(anchor.y1 + ya * yRatio);

                    // Anchors width and height
                    const int wa = xa1 - xa0;
                    const int ha = ya1 - ya0;

                    // Anchor center coordinates (xac, yac)
                    const float xac = xa0 + wa / 2.0;
                    const float yac = ya0 + ha / 2.0;

                    /**
                     * 1st condition: "During  training,  we  ignore all
                     * cross-boundary anchors so they do not contribute to  the
                     * loss."
                     * 2nd condition: "During testing, however, we still apply
                     * the fully convolutional RPN  to  the  entire  image."
                    */
                    // Score
                    const unsigned int clsIdx = xa + ya*outputWidth
                                                + k*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int txIdx = xa + ya*outputWidth
                                                + (k + scoreCls*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int tyIdx = xa + ya*outputWidth
                                                + (k + (scoreCls + 1)*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int twIdx = xa + ya*outputWidth
                                                + (k + (scoreCls + 2)*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int thIdx = xa + ya*outputWidth
                                                + (k + (scoreCls + 3)*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;;

                    const float cls = inputs[clsIdx];

                    // Parameterized coordinates
                    const float txbb = inputs[txIdx];

                    const float tybb = inputs[tyIdx];

                    const float twbb = inputs[twIdx];

                    const float thbb = inputs[thIdx];

                    // Predicted box center coordinates
                    const float xbbc = ((isFlip) ? -txbb : txbb) * wa
                                            + xac;
                    const float ybbc = ((isFlip) ? -tybb : tybb) * ha
                                            + yac;
                    float wbb = wa * std::exp(twbb);
                    float hbb = ha * std::exp(thbb);

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
                    if (xbb + wbb > stimuliWidth - 1)
                        wbb = stimuliWidth - 1 - xbb;
                    if (ybb + hbb > stimuliHeight - 1)
                        hbb = stimuliHeight - 1 - ybb;

                    const unsigned int OutClsIdx = xa + ya*outputWidth
                                                + k*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int xIdx = xa + ya*outputWidth
                                                + (k + 1*nbAnchors) *outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int yIdx = xa + ya*outputWidth
                                                + (k + 2*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int wIdx = xa + ya*outputWidth
                                                + (k + 3*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int hIdx = xa + ya*outputWidth
                                                + (k + 4*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    const unsigned int argIdx = xa + ya*outputWidth
                                                + (k + 5*nbAnchors)*outputWidth*outputHeight
                                                + batchPos*nbAnchors*outputHeight*outputWidth;
                    outputs[OutClsIdx] = cls;
                    outputs[xIdx] = xbb;
                    outputs[yIdx] = ybb;
                    outputs[wIdx] = wbb;
                    outputs[hIdx] = hbb;
                    outputs[argIdx] = 0.0;
                }
            }
        }
    }

}

void region_proposal_cpu(unsigned int batchSize,
                         unsigned int nbOutputs,
                         unsigned int outputHeight,
                         unsigned int outputWidth,
                         unsigned int nbAnchors,
                         unsigned int channelHeight,
                         unsigned int channelWidth,
                         unsigned int nbProposals,
                         unsigned int preNmsTopN,
                         double nmsIoU,
                         double minHeight,
                         double minWidth,
                         unsigned int scoreIndex,
                         unsigned int iouIndex,
                         const float* inputs,
                         float* outputs)
{
    for (int batchPos = 0; batchPos < (int)batchSize; ++batchPos) {
        // Collect all ROIs in the "ROIs" vector
        std::vector<std::pair<ROI, float> > ROIs;
        ROIs.reserve(nbAnchors * channelHeight * channelWidth);

        for (unsigned int k = 0; k < nbAnchors; ++k) {
            for (unsigned int y = 0; y < channelHeight; ++y) {
                for (unsigned int x = 0; x < channelWidth; ++x) {
                    const unsigned int vIdx = x + y*channelWidth
                                                + (k + scoreIndex*nbAnchors)*channelWidth*channelHeight
                                                + batchPos*nbAnchors*channelWidth*channelHeight;
                    const unsigned int wIdx = x + y*channelWidth
                                                + (k + 3*nbAnchors)*channelWidth*channelHeight
                                                + batchPos*nbAnchors*channelWidth*channelHeight;
                    const unsigned int hIdx = x + y*channelWidth
                                                + (k + 4*nbAnchors)*channelWidth*channelHeight
                                                + batchPos*nbAnchors*channelWidth*channelHeight;

                    const float value = inputs[vIdx];
                    const float w = inputs[wIdx];
                    const float h = inputs[hIdx];

                    if (value >= 0.0 && w >= minWidth && h >= minHeight) {
                        ROIs.push_back(std::make_pair(ROI(x, y, k, batchPos), value));
                    }
                }
            }
        }

        // Sort ROIs by value
        if (preNmsTopN > 0 && preNmsTopN < ROIs.size()) {
            std::partial_sort(ROIs.begin(),
                              ROIs.begin() + preNmsTopN,
                              ROIs.end(),
                              PairSecondPred<ROI, float, std::greater<float> >());

            // Drop the lowest score (unsorted) ROIs
            ROIs.resize(preNmsTopN);
        }
        else {
            std::sort(ROIs.begin(),
                      ROIs.end(),
                      PairSecondPred<ROI, float, std::greater<float> >());
        }

        // Non-Maximum Suppression (NMS)
        for (unsigned int i = 0; i < ROIs.size() - 1 && i < nbProposals;
            ++i)
        {
            const ROI& ROIMax = ROIs[i].first;

            unsigned int
                xIdx = ROIMax.i + ROIMax.j*channelWidth
                        +(ROIMax.k + nbAnchors)*channelWidth*channelHeight
                        +ROIMax.b*nbAnchors*channelWidth*channelHeight;
            unsigned int
                yIdx = ROIMax.i + ROIMax.j*channelWidth
                        +(ROIMax.k + 2*nbAnchors)*channelWidth*channelHeight
                        + ROIMax.b*nbAnchors*channelWidth*channelHeight;
            unsigned int
                wIdx = ROIMax.i + ROIMax.j*channelWidth
                      + (ROIMax.k + 3*nbAnchors)*channelWidth*channelHeight
                      + ROIMax.b*nbAnchors*channelWidth*channelHeight;
            unsigned int
                hIdx = ROIMax.i + ROIMax.j*channelWidth
                       + (ROIMax.k + 4*nbAnchors)*channelWidth*channelHeight
                       + ROIMax.b*nbAnchors*channelWidth*channelHeight;

            const float x0 = inputs[xIdx];
            const float y0 = inputs[yIdx];
            const float w0 = inputs[wIdx];
            const float h0 = inputs[hIdx];


            for (unsigned int j = i + 1; j < ROIs.size(); ) {
                const ROI& ROI = ROIs[j].first;

                xIdx = ROI.i + ROI.j*channelWidth
                        + (ROI.k + nbAnchors)*channelWidth*channelHeight
                        + ROI.b*nbAnchors*channelWidth*channelHeight;
                yIdx = ROI.i + ROI.j*channelWidth
                        + (ROI.k + 2*nbAnchors)*channelWidth*channelHeight
                        + ROI.b*nbAnchors*channelWidth*channelHeight;
                wIdx = ROI.i + ROI.j*channelWidth
                        + (ROI.k + 3*nbAnchors)*channelWidth*channelHeight
                        + ROI.b*nbAnchors*channelWidth*channelHeight;
                hIdx = ROI.i + ROI.j*channelWidth
                        + (ROI.k + 4*nbAnchors)*channelWidth*channelHeight
                        + ROI.b*nbAnchors*channelWidth*channelHeight;

                const float x = inputs[xIdx];
                const float y = inputs[yIdx];
                const float w = inputs[wIdx];
                const float h = inputs[hIdx];

                const float interLeft = std::max(x0, x);
                const float interRight = std::min(x0 + w0, x + w);
                const float interTop = std::max(y0, y);
                const float interBottom = std::min(y0 + h0, y + h);

                if (interLeft < interRight && interTop < interBottom) {
                    const float interArea = (interRight - interLeft)
                                                * (interBottom - interTop);
                    const float unionArea = w0 * h0 + w * h - interArea;
                    const float IoU = interArea / unionArea;

                    if (IoU > nmsIoU) {
                        // Suppress ROI
                        ROIs.erase(ROIs.begin() + j);
                        continue;
                    }
                }

                ++j;
            }
        }

        ROIs.resize(nbProposals);

        // Keep the top-N ROIs
        for (unsigned int n = 0; n < nbProposals; ++n) {
            const unsigned int
                xIdx = ROIs[n].first.i + ROIs[n].first.j*channelWidth
                        +(ROIs[n].first.k + nbAnchors)*channelWidth*channelHeight
                        +ROIs[n].first.b*nbAnchors*channelWidth*channelHeight;
            const unsigned int
                yIdx = ROIs[n].first.i + ROIs[n].first.j*channelWidth
                        +(ROIs[n].first.k + 2*nbAnchors)*channelWidth*channelHeight
                        +ROIs[n].first.b*nbAnchors*channelWidth*channelHeight;
            const unsigned int
                wIdx = ROIs[n].first.i + ROIs[n].first.j*channelWidth
                      + (ROIs[n].first.k + 3*nbAnchors)*channelWidth*channelHeight
                      +ROIs[n].first.b*nbAnchors*channelWidth*channelHeight;
            const unsigned int
                hIdx = ROIs[n].first.i + ROIs[n].first.j*channelWidth
                       + (ROIs[n].first.k + 4*nbAnchors)*channelWidth*channelHeight
                       +ROIs[n].first.b*nbAnchors*channelWidth*channelHeight;

            outputs[0 + (n + batchPos*nbProposals)*4]
                = inputs[xIdx];
            outputs[1 + (n + batchPos*nbProposals)*4]
                = inputs[yIdx];
            outputs[2 + (n + batchPos*nbProposals)*4]
                = inputs[wIdx];
            outputs[3 + (n + batchPos*nbProposals)*4]
                = inputs[hIdx];
        }
    }
}

void ROIPooling_bilinear_cpu(unsigned int batchSize,
                             unsigned int nbOutputs,
                             unsigned int outputHeight,
                             unsigned int outputWidth,
                             unsigned int stimuliHeight,
                             unsigned int stimuliWidth,
                             std::vector<trt_Dims3> featureDims,
                             unsigned int nbProposals,
                             const float* inputs,
                             float* outputs)
{
    const float alpha = 1.0f;
    float beta = 0.0f;

    unsigned int outputOffset = 0;
    unsigned int inputFeatureOffset = nbProposals*4;

    for (unsigned int k = 0, size = featureDims.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const double yRatio = std::ceil(stimuliHeight
                                        / (double)featureDims[k].d[1]);
        const double xRatio = std::ceil(stimuliWidth
                                        / (double)featureDims[k].d[2]);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) /*if (size > 16)*/
#else
#pragma omp parallel for if (batchSize > 4 /*&& size > 16*/)
#endif
            for (int batchPos = 0; batchPos < (int)batchSize; ++batchPos)
            {
                for (unsigned int channel = 0; channel < featureDims[k].d[0];
                    ++channel)
                {
                    const unsigned int inputBatch = batchPos / nbProposals;

                    float x = (inputs[0 + batchPos*4] - 1.0) / xRatio;
                    float y = (inputs[1 + batchPos*4] - 6.0) / yRatio;
                    float w = (inputs[2 + batchPos*4] / xRatio);
                    float h = (inputs[3 + batchPos*4] / yRatio);

                    // Crop ROI to image boundaries
                    if (x < 0) {
                        w+= x;
                        x = 0;
                    }
                    if (y < 0) {
                        h+= y;
                        y = 0;
                    }
                    if (x + w > (int)featureDims[k].d[2])
                        w = featureDims[k].d[2] - x;
                    if (y + h > (int)featureDims[k].d[1])
                        h = featureDims[k].d[1] - y;

                    const float yPoolRatio = h / (outputHeight - 1);
                    const float xPoolRatio = w / (outputWidth - 1);

                    for (unsigned int oy = 0; oy < outputHeight; ++oy) {
                        for (unsigned int ox = 0; ox < outputWidth; ++ox) {
                            // -0.5 + (ox + 0.5) and not ox because the
                            // interpolation is done relative to the CENTER of
                            // the pixels
                            const float sy = clamp_export<float>(y + oy * yPoolRatio, 0, featureDims[k].d[1] - 1);
                            const float sx = clamp_export<float>(x + ox * xPoolRatio, 0, featureDims[k].d[2] - 1);

                            const unsigned int sx0 = (int)(sx);
                            const unsigned int sy0 = (int)(sy);

                            const float dx = sx - sx0;
                            const float dy = sy - sy0;


                            const unsigned int idxI00 = sx0 + sy0*featureDims[k].d[2]
                                                        + channel*featureDims[k].d[1]*featureDims[k].d[2]
                                                        + inputFeatureOffset
                                                        + inputBatch
                                                            *(inputFeatureOffset
                                                            + featureDims[k].d[0]
                                                                * featureDims[k].d[1]
                                                                * featureDims[k].d[2]);

                            const unsigned int idxI10 = (sx0 + 1) + sy0*featureDims[k].d[2]
                                                        + channel*featureDims[k].d[1]*featureDims[k].d[2]
                                                        + inputFeatureOffset
                                                        + inputBatch
                                                            *(inputFeatureOffset
                                                            + featureDims[k].d[0]
                                                                * featureDims[k].d[1]
                                                                * featureDims[k].d[2]);

                            const unsigned int idxI01 = sx0+ (sy0 + 1)*featureDims[k].d[2]
                                                        + channel*featureDims[k].d[1]*featureDims[k].d[2]
                                                        + inputFeatureOffset
                                                        + inputBatch
                                                            *(inputFeatureOffset
                                                            + featureDims[k].d[0]
                                                                * featureDims[k].d[1]
                                                                * featureDims[k].d[2]);

                            const unsigned int idxI11 = (sx0 + 1) + (sy0 + 1)*featureDims[k].d[2]
                                                        + channel*featureDims[k].d[1]*featureDims[k].d[2]
                                                        + inputFeatureOffset
                                                        + inputBatch
                                                            *(inputFeatureOffset
                                                            + featureDims[k].d[0]
                                                                * featureDims[k].d[1]
                                                                * featureDims[k].d[2]);


                            const float i00 = inputs[idxI00];

                            const float i10 = (sx0 + 1 < featureDims[k].d[2]) ?
                                                 inputs[idxI10] : 0.0;

                            const float i01 = (sy0 + 1 < featureDims[k].d[1]) ?
                                                 inputs[idxI01]: 0.0;

                            const float i11 = (sx0 + 1 < featureDims[k].d[2]
                                                 && sy0 + 1 < featureDims[k].d[1])
                                                 ? inputs[idxI11] : 0.0;

                            const float value
                                = i00 * (1 - dx) * (1 - dy)
                                + i10 * dx * (1 - dy)
                                + i01 * (1 - dx) * dy
                                + i11 * (dx * dy);

                            const unsigned int outIdx = ox + oy*outputWidth
                                                        + (outputOffset + channel)*outputWidth*outputHeight
                                                        +  batchPos*nbOutputs*outputWidth*outputHeight;

                            outputs[outIdx] = alpha * value + beta *outputs[outIdx];
                        }
                    }
                }
            }
        outputOffset += featureDims[k].d[0];
        inputFeatureOffset += featureDims[k].d[0]
                                *featureDims[k].d[1]
                                *featureDims[k].d[2];
    }
}


void object_det_cpu(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int channelHeight,
                    unsigned int channelWidth,
                    unsigned int nbAnchors,
                    unsigned int nbProposals,
                    unsigned int nbClass,
                    double nmsIoU,
                    const float* scoreThreshold,
                    const float* inputs,
                    float* outputs)
{

    const int inputBatch = batchSize/(nbAnchors*nbClass);

    std::vector< std::vector< std::vector<BBox_T> > > ROIs;
    ROIs.resize(inputBatch);

    for (int batchPos = 0; batchPos < inputBatch; ++batchPos) {

        ROIs[batchPos].resize(nbClass);
        unsigned int nbRoiDetected = 0;

        for(unsigned int cls = 0; cls < nbClass; ++ cls)
        {
            for (unsigned int anchor = 0; anchor < nbAnchors; ++anchor)
            {
                for (unsigned int y = 0; y < channelHeight; ++y) {
                    for (unsigned int x = 0; x < channelWidth; ++x) {

                        const float value = inputs[  x
                                                    + y*channelWidth
                                                    + (anchor + cls*nbAnchors)*channelWidth*channelHeight
                                                    + batchPos*channelWidth*channelHeight*nbClass*nbAnchors*6];

                        if(value >= scoreThreshold[cls])
                        {
                            const unsigned int offset = nbAnchors*nbClass*channelWidth*channelHeight;

                            const float xbbEst = inputs[ x
                                                        + y*channelWidth
                                                        + (anchor + cls*nbAnchors)*channelHeight*channelWidth
                                                        + offset
                                                        + batchPos*channelWidth*channelHeight*nbClass*nbAnchors*6];

                            const float ybbEst = inputs[ x
                                                        + y*channelWidth
                                                        + (anchor + cls*nbAnchors)*channelHeight*channelWidth
                                                        + 2*offset
                                                        + batchPos*channelWidth*channelHeight*nbClass*nbAnchors*6];
                            const float wbbEst = inputs[ x
                                                        + y*channelWidth
                                                        + (anchor + cls*nbAnchors)*channelHeight*channelWidth
                                                        + 3*offset
                                                        + batchPos*channelWidth*channelHeight*nbClass*nbAnchors*6];

                            const float hbbEst = inputs[ x
                                                        + y*channelWidth
                                                        + (anchor + cls*nbAnchors)*channelHeight*channelWidth
                                                        + 4*offset
                                                        + batchPos*channelWidth*channelHeight*nbClass*nbAnchors*6];

                            ROIs[batchPos][cls].push_back(BBox_T(xbbEst,ybbEst,wbbEst,hbbEst, 0.0));
                        }

                    }
                }

            }

            if(ROIs[batchPos][cls].size() > 0)
            {
                // Non-Maximum Suppression (NMS)
                for (unsigned int i = 0; i < ROIs[batchPos][cls].size() - 1; ++i)
                {
                    const float x0 = ROIs[batchPos][cls][i].x;
                    const float y0 = ROIs[batchPos][cls][i].y;
                    const float w0 = ROIs[batchPos][cls][i].w;
                    const float h0 = ROIs[batchPos][cls][i].h;

                    for (unsigned int j = i + 1; j < ROIs[batchPos][cls].size(); ) {

                        const float x = ROIs[batchPos][cls][j].x;
                        const float y = ROIs[batchPos][cls][j].y;
                        const float w = ROIs[batchPos][cls][j].w;
                        const float h = ROIs[batchPos][cls][j].h;

                        const float interLeft = std::max(x0, x);
                        const float interRight = std::min(x0 + w0, x + w);
                        const float interTop = std::max(y0, y);
                        const float interBottom = std::min(y0 + h0, y + h);

                        if (interLeft < interRight && interTop < interBottom) {
                            const float interArea = (interRight - interLeft)
                                                        * (interBottom - interTop);
                            const float unionArea = w0 * h0 + w * h - interArea;
                            const float IoU = interArea / unionArea;

                            if (IoU > nmsIoU) {
                                // Suppress ROI
                                ROIs[batchPos][cls].erase(ROIs[batchPos][cls].begin() + j);
                                continue;
                            }
                        }
                        ++j;
                    }
                }
            }

            nbRoiDetected += ROIs[batchPos][cls].size();

        }
        for (unsigned int cls = 0; cls < nbClass; ++cls)
        {
            unsigned int totalIdxPerClass = 0;

            for(unsigned int i = 0; i < ROIs[batchPos][cls].size() && totalIdxPerClass < nbProposals; ++i)
            {
                const unsigned int n = totalIdxPerClass + cls*nbProposals + batchPos*nbProposals*nbClass;
                outputs[0 + n*5] = ROIs[batchPos][cls][i].x;
                outputs[1 + n*5] = ROIs[batchPos][cls][i].y;
                outputs[2 + n*5] = ROIs[batchPos][cls][i].w;
                outputs[3 + n*5] = ROIs[batchPos][cls][i].h;
                outputs[4 + n*5] = (float) cls;

                totalIdxPerClass++;
            }

            for(unsigned int rest = totalIdxPerClass; rest < nbProposals; ++rest)
            {
                    const unsigned int n = rest + cls*nbProposals +  batchPos*nbProposals*nbClass;
                    outputs[0 + n*5] = 0.0;
                    outputs[1 + n*5] = 0.0;
                    outputs[2 + n*5] = 0.0;
                    outputs[3 + n*5] = 0.0;
                    outputs[4 + n*5] = 0.0;
            }
        }

    }
}
