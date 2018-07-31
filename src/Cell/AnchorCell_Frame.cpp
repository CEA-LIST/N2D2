/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#include "Cell/AnchorCell_Frame.hpp"

N2D2::Registrar<N2D2::AnchorCell>
N2D2::AnchorCell_Frame::mRegistrar("Frame", N2D2::AnchorCell_Frame::create);

N2D2::AnchorCell_Frame::AnchorCell_Frame(
    const std::string& name,
    StimuliProvider& sp,
    const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
    unsigned int scoresCls)
    : Cell(name, 6*anchors.size()),
      AnchorCell(name, sp, anchors, scoresCls),
      Cell_Frame(name, 6*anchors.size()),
      mAnchors(anchors)
{
    // ctor
}
int N2D2::AnchorCell_Frame::getNbAnchors() const { return mAnchors.size(); }

std::vector<N2D2::Float_T> N2D2::AnchorCell_Frame::getAnchor(unsigned int idx)
const
{
    std::vector<Float_T> vect_anchor;
    vect_anchor.push_back(mAnchors[idx].x0);
    vect_anchor.push_back(mAnchors[idx].y0);
    vect_anchor.push_back(mAnchors[idx].x1);
    vect_anchor.push_back(mAnchors[idx].y1);
    return vect_anchor;
}

const std::vector<N2D2::AnchorCell_Frame_Kernels::BBox_T>&
N2D2::AnchorCell_Frame::getGT(unsigned int batchPos) const
{
    assert(batchPos < mGT.size());

    return mGT[batchPos];
}

std::shared_ptr<N2D2::ROI>
N2D2::AnchorCell_Frame::getAnchorROI(const Tensor<int>::Index& index) const
{
    const int argMaxIoU = mArgMaxIoU(index);

    if (argMaxIoU >= 0) {
        std::vector<std::shared_ptr<ROI> > labelROIs
            = mStimuliProvider.getLabelsROIs(index[3]);
        assert(argMaxIoU < (int)labelROIs.size());
        return labelROIs[argMaxIoU];
    }
    else
        return std::shared_ptr<ROI>();
}

N2D2::AnchorCell_Frame_Kernels::BBox_T
N2D2::AnchorCell_Frame::getAnchorBBox(const Tensor<int>::Index& index) const
{
    assert(index[0] < mArgMaxIoU.dimX());
    assert(index[1] < mArgMaxIoU.dimY());
    assert(index[2] < mArgMaxIoU.dimZ());
    assert(index[3] < mArgMaxIoU.dimB());

    const unsigned int nbAnchors = mAnchors.size();
    const Float_T xa = mOutputs(index[0], index[1],
                                index[2] + 1 * nbAnchors, index[3]);
    const Float_T ya = mOutputs(index[0], index[1],
                                index[2] + 2 * nbAnchors, index[3]);
    const Float_T wa = mOutputs(index[0], index[1],
                                index[2] + 3 * nbAnchors, index[3]);
    const Float_T ha = mOutputs(index[0], index[1],
                                index[2] + 4 * nbAnchors, index[3]);
    return AnchorCell_Frame_Kernels::BBox_T(xa, ya, wa, ha);
}

N2D2::AnchorCell_Frame_Kernels::BBox_T
N2D2::AnchorCell_Frame::getAnchorGT(const Tensor<int>::Index& index) const
{
    assert(index[3] < mGT.size());

    const int argMaxIoU = mArgMaxIoU(index);

    assert(argMaxIoU < 0 || argMaxIoU < (int)mGT[index[3]].size());

    return ((argMaxIoU >= 0)
        ? mGT[index[3]][argMaxIoU]
        : AnchorCell_Frame_Kernels::BBox_T(0.0, 0.0, 0.0, 0.0));
}

N2D2::Float_T
N2D2::AnchorCell_Frame::getAnchorIoU(const Tensor<int>::Index& index) const
{
    assert(index[0] < mArgMaxIoU.dimX());
    assert(index[1] < mArgMaxIoU.dimY());
    assert(index[2] < mArgMaxIoU.dimZ());
    assert(index[3] < mArgMaxIoU.dimB());

    return mOutputs(index[0], index[1], index[2] + 5 * mAnchors.size(), index[3]);
}

int
N2D2::AnchorCell_Frame::getAnchorArgMaxIoU(const Tensor<int>::Index& index)
    const
{
    return mArgMaxIoU(index);
}

void N2D2::AnchorCell_Frame::initialize()
{
    const unsigned int nbAnchors = mAnchors.size();
    if(mFeatureMapWidth == 0)
        mFeatureMapWidth = mStimuliProvider.getSizeX();

    if(mFeatureMapHeight == 0)
        mFeatureMapHeight = mStimuliProvider.getSizeY();

    if (mInputs.dimZ() != (mScoresCls + 5) * nbAnchors) {
        throw std::domain_error("AnchorCell_Frame::initialize():"
                                " the number of input channels must be equal to"
                                " (scoresCls + 5) times the number of"
                                " anchors.");
    }

    if (mFlip) {
        const double xRatio = std::ceil(mStimuliProvider.getSizeX()
                                        / (double)mOutputsDims[0]);
        const double yRatio = std::ceil(mStimuliProvider.getSizeY()
                                        / (double)mOutputsDims[1]);

        const double xOffset = mStimuliProvider.getSizeX() - 1
                                - (mOutputsDims[0] - 1) * xRatio;
        const double yOffset = mStimuliProvider.getSizeY() - 1
                                - (mOutputsDims[1] - 1) * yRatio;

        for (unsigned int k = 0; k < nbAnchors; ++k) {
            AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors[k];
            anchor.x0 += xOffset;
            anchor.y0 += yOffset;
            anchor.x1 += xOffset;
            anchor.y1 += yOffset;
        }
    }

    mGT.resize(mOutputs.dimB());
    mArgMaxIoU.resize({mOutputsDims[0],
                      mOutputsDims[1],
                      mAnchors.size(),
                      mOutputs.dimB()});
}

void N2D2::AnchorCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDToH();

    const Tensor<Float_T>& inputsCls = tensor_cast<Float_T>(mInputs[0]);
    const Tensor<Float_T>& inputsCoords = (mInputs.size() > 1)
        ? tensor_cast<Float_T>(mInputs[1]) : inputsCls;
    const unsigned int coordsOffset = (mInputs.size() > 1)
        ? 0 : mScoresCls;

    const unsigned int nbAnchors = mAnchors.size();
    const double xRatio = std::ceil(mFeatureMapWidth
                                    / (double)mOutputsDims[0]);
    const double yRatio = std::ceil(mFeatureMapHeight
                                    / (double)mOutputsDims[1]);

    const float xOutputRatio = mStimuliProvider.getSizeX() 
                                    / (float) mFeatureMapWidth;
    const float yOutputRatio = mStimuliProvider.getSizeY() 
                                    / (float) mFeatureMapHeight;

#pragma omp parallel for if (mOutputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
        std::vector<AnchorCell_Frame_Kernels::BBox_T>& GT = mGT[batchPos];

        // Extract ground true ROIs
        std::vector<std::shared_ptr<ROI> > labelROIs
            = mStimuliProvider.getLabelsROIs(batchPos);

        GT.resize(labelROIs.size());

        for (unsigned int l = 0; l < labelROIs.size(); ++l) {
            cv::Rect labelRect = labelROIs[l]->getBoundingRect();

            // Crop labelRect to the slice for correct overlap area calculation
            if (labelRect.tl().x < 0) {
                labelRect.width+= labelRect.tl().x;
                labelRect.x = 0;
            }
            if (labelRect.tl().y < 0) {
                labelRect.height+= labelRect.tl().y;
                labelRect.y = 0;
            }
            if (labelRect.br().x > (int)mStimuliProvider.getSizeX())
                labelRect.width = mStimuliProvider.getSizeX() - labelRect.x;
            if (labelRect.br().y > (int)mStimuliProvider.getSizeY())
                labelRect.height = mStimuliProvider.getSizeY() - labelRect.y;

            GT[l] = AnchorCell_Frame_Kernels::BBox_T(labelRect.tl().x,
                                                     labelRect.tl().y,
                                                     labelRect.width,
                                                     labelRect.height);
        }
    }

    const unsigned int size = mOutputs.dimB() * nbAnchors;

    mMaxIoU.assign(mOutputs.dimB(), 0.0);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mOutputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
        for (unsigned int k = 0; k < nbAnchors; ++k) {
            const std::vector<AnchorCell_Frame_Kernels::BBox_T>& GT
                = mGT[batchPos];
            const AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors[k];
/*
            // DEBUG
            std::cout << "Number of GT boxes: " << GT.size() << std::endl;

            for (unsigned int i = 0, size = GT.size(); i < size; ++i) {
                int xgt, ygt, wgt, hgt;
                std::tie(xgt, ygt, wgt, hgt) = GT[i];

                std::cout << "  " << wgt << "x" << hgt
                    << " @ (" << xgt << "," << ygt << ")" << std::endl;
            }

            cv::Mat imgIoUHsv(cv::Size(mOutputsDims[0], mOutputsDims[1]),
                                    CV_8UC3,
                                    cv::Scalar(0, 0, 0));
            cv::Mat imgClsHsv(cv::Size(mOutputsDims[0], mOutputsDims[1]),
                                    CV_8UC3,
                                    cv::Scalar(0, 0, 0));
*/
            for (unsigned int ya = 0; ya < mOutputsDims[1]; ++ya) {
                for (unsigned int xa = 0; xa < mOutputsDims[0]; ++xa) {
                    // Shifted anchors coordinates at (xa, ya)
                    const int xa0 = (int)(anchor.x0 + xa * xRatio);
                    const int ya0 = (int)(anchor.y0 + ya * yRatio);
                    const int xa1 = (int)(anchor.x1 + xa * xRatio);
                    const int ya1 = (int)(anchor.y1 + ya * yRatio);

                    // Anchors width and height
                    const int wa = xa1 - xa0;
                    const int ha = ya1 - ya0;

                    // Anchor center coordinates (xac, yac)
                    const Float_T xac = xa0 + wa / 2.0;
                    const Float_T yac = ya0 + ha / 2.0;

                    /**
                     * 1st condition: "During  training,  we  ignore all
                     * cross-boundary anchors so they do not contribute to  the
                     * loss."
                     * 2nd condition: "During testing, however, we still apply
                     * the fully convolutional RPN  to  the  entire  image."
                    */
                    if ((xa0 >= 0
                        && ya0 >= 0
                        && xa1 < (int)mFeatureMapWidth
                        && ya1 < (int)mFeatureMapHeight)
                        || inference)
                    {
                        // Score
                        const Float_T cls = inputsCls(xa, ya, k, batchPos);

                        // Parameterized coordinates
                        const Float_T txbb = inputsCoords(xa, ya,
                            k + coordsOffset * nbAnchors, batchPos);
                        const Float_T tybb = inputsCoords(xa, ya,
                            k + (coordsOffset + 1) * nbAnchors, batchPos);
                        const Float_T twbb = inputsCoords(xa, ya,
                            k + (coordsOffset + 2) * nbAnchors, batchPos);
                        const Float_T thbb = inputsCoords(xa, ya,
                            k + (coordsOffset + 3) * nbAnchors, batchPos);

                        // Predicted box center coordinates
                        const Float_T xbbc = ((mFlip) ? -txbb : txbb) * wa
                                                + xac;
                        const Float_T ybbc = ((mFlip) ? -tybb : tybb) * ha
                                                + yac;
                        Float_T wbb = wa * std::exp(twbb);
                        Float_T hbb = ha * std::exp(thbb);

                        // Predicted box top-left coordinates
                        Float_T xbb = xbbc - wbb / 2.0;
                        Float_T ybb = ybbc - hbb / 2.0;

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
                            if (xbb + wbb > mFeatureMapWidth - 1)
                                wbb = mFeatureMapWidth - 1 - xbb;
                            if (ybb + hbb > mFeatureMapHeight - 1)
                                hbb = mFeatureMapHeight - 1 - ybb;
                        }

                        // For inference, compute IoU on predicted boxes
                        // For learning, compute IoU on anchor boxes
                        // => if IoU is computed on predicted boxes during
                        // learning, predicted boxes may arbitrarily drift from
                        // anchors and learning does not converge
                        const AnchorCell_Frame_Kernels::BBox_T bb = (inference)
                            ? AnchorCell_Frame_Kernels::BBox_T
                                (xbb, ybb, wbb, hbb)
                            : AnchorCell_Frame_Kernels::BBox_T
                                (xa0, ya0, wa, ha);

                        Float_T maxIoU = 0.0;
                        int argMaxIoU = -1;

                        for (unsigned int l = 0, nbLabels = GT.size();
                            l < nbLabels; ++l)
                        {
                            // Ground Truth box coordinates
                            const AnchorCell_Frame_Kernels::BBox_T& gt = GT[l];

                            const Float_T interLeft = std::max(gt.x, bb.x);
                            const Float_T interRight = std::min(gt.x + gt.w,
                                                                bb.x + bb.w);
                            const Float_T interTop = std::max(gt.y, bb.y);
                            const Float_T interBottom = std::min(gt.y + gt.h,
                                                                 bb.y + bb.h);

                            if (interLeft < interRight
                                && interTop < interBottom)
                            {
                                const Float_T interArea
                                    = (interRight - interLeft)
                                        * (interBottom - interTop);
                                const Float_T unionArea = gt.w * gt.h
                                    + bb.w * bb.h - interArea;
                                const Float_T IoU = interArea / unionArea;

                                if (IoU > maxIoU) {
                                    maxIoU = IoU;
                                    argMaxIoU = l;
                                }
                            }
                        }

                        //Rescale Bounding Box if Feature MAP size is different than stimuli size
                        xbb *=  xOutputRatio;
                        wbb *=  xOutputRatio;
                        ybb *=  yOutputRatio;
                        hbb *=  yOutputRatio;

                        mOutputs(xa, ya, k, batchPos) = cls;
                        mOutputs(xa, ya, k + 1 * nbAnchors, batchPos) = xbb;
                        mOutputs(xa, ya, k + 2 * nbAnchors, batchPos) = ybb;
                        mOutputs(xa, ya, k + 3 * nbAnchors, batchPos) = wbb;
                        mOutputs(xa, ya, k + 4 * nbAnchors, batchPos) = hbb;
                        mOutputs(xa, ya, k + 5 * nbAnchors, batchPos) = maxIoU;
                        mArgMaxIoU(xa, ya, k, batchPos) = argMaxIoU;
                        mMaxIoU[batchPos] = std::max(mMaxIoU[batchPos], maxIoU);
/*
                        // DEBUG
                        if (batchPos == 0) {
                            imgIoUHsv.at<cv::Vec3b>(ya, xa)
                                = cv::Vec3f(0, 255, 255 * maxIoU);
                            imgClsHsv.at<cv::Vec3b>(ya, xa)
                                = cv::Vec3f(0, 255, 255 * cls);
                        }
*/
                    }
                    else {
                        mOutputs(xa, ya, k, batchPos) = -1.0;
                        mOutputs(xa, ya, k + 1 * nbAnchors, batchPos) = 0.0;
                        mOutputs(xa, ya, k + 2 * nbAnchors, batchPos) = 0.0;
                        mOutputs(xa, ya, k + 3 * nbAnchors, batchPos) = 0.0;
                        mOutputs(xa, ya, k + 4 * nbAnchors, batchPos) = 0.0;
                        mOutputs(xa, ya, k + 5 * nbAnchors, batchPos) = 0.0;
                        mArgMaxIoU(xa, ya, k, batchPos) = -1;
                    }
                }
            }
/*
            // DEBUG
            if (batchPos == 0) {
                const double alpha = 0.25;
                Utils::createDirectories(mName);

                // Input image
                cv::Mat img = (cv::Mat)mStimuliProvider.getData(0, batchPos);
                cv::Mat img8U;
                // img.convertTo(img8U, CV_8U, 255.0);

                // Normalize image
                cv::Mat imgNorm;
                cv::normalize(img.reshape(1), imgNorm, 0, 255, cv::NORM_MINMAX);
                img = imgNorm.reshape(img.channels());
                img.convertTo(img8U, CV_8U);

                cv::Mat imgColor;
                cv::cvtColor(img8U, imgColor, CV_GRAY2BGR);

                for (unsigned int l = 0, nbLabels = GT.size(); l < nbLabels;
                    ++l)
                {
                    Float_T xgt, ygt, wgt, hgt;
                    std::tie(xgt, ygt, wgt, hgt) = GT[l];

                    cv::rectangle(imgColor,
                                  cv::Point(xgt, ygt),
                                  cv::Point(xgt + wgt, ygt + hgt),
                                  cv::Scalar(255, 0, 0));
                }

                for (unsigned int ya = 0; ya < mOutputsDims[1]; ++ya) {
                    for (unsigned int xa = 0; xa < mOutputsDims[0]; ++xa) {
                        const Float_T x = mOutputs(xa, ya,
                                                   k + 1 * nbAnchors, batchPos);
                        const Float_T y = mOutputs(xa, ya,
                                                   k + 2 * nbAnchors, batchPos);
                        const Float_T w = mOutputs(xa, ya,
                                                   k + 3 * nbAnchors, batchPos);
                        const Float_T h = mOutputs(xa, ya,
                                                   k + 4 * nbAnchors, batchPos);

                        // Eliminate cross-boundary anchors
                        if (x >= 0.0
                            && y >= 0.0
                            && (x + w) < mStimuliProvider.getSizeX()
                            && (y + h) < mStimuliProvider.getSizeY())
                        {
                            const Float_T IoU = mOutputs(xa, ya,
                                                k + 5 * nbAnchors, batchPos);

                            if (IoU > mPositiveIoU) {
                                cv::rectangle(imgColor,
                                              cv::Point(x, y),
                                              cv::Point(x + w, y + h),
                                              cv::Scalar(0, 0, 255));
                            }
                        }
                    }
                }

                // Target image IoU
                cv::Mat imgIoU;
                cv::cvtColor(imgIoUHsv, imgIoU, CV_HSV2BGR);

                cv::Mat imgSampled;
                cv::resize(imgIoU,
                           imgSampled,
                           cv::Size(mStimuliProvider.getSizeX(),
                                    mStimuliProvider.getSizeY()),
                           0.0,
                           0.0,
                           cv::INTER_NEAREST);

                cv::Mat imgBlended;
                cv::addWeighted(
                    imgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);

                std::string fileName = mName + "/anchor_"
                                        + std::to_string(k) + ".png";

                if (!cv::imwrite(fileName, imgBlended))
                    throw std::runtime_error("Unable to write image: "
                                             + fileName);

                // Target image Cls
                cv::Mat imgCls;
                cv::cvtColor(imgClsHsv, imgCls, CV_HSV2BGR);

                cv::resize(imgCls,
                           imgSampled,
                           cv::Size(mStimuliProvider.getSizeX(),
                                    mStimuliProvider.getSizeY()),
                           0.0,
                           0.0,
                           cv::INTER_NEAREST);

                cv::addWeighted(
                    imgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);

                fileName = mName + "/anchor_"
                                        + std::to_string(k) + "_cls.png";

                if (!cv::imwrite(fileName, imgBlended))
                    throw std::runtime_error("Unable to write image: "
                                             + fileName);
            }
*/
        }
    }

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::AnchorCell_Frame::backPropagate()
{
    const unsigned int nbAnchors = mAnchors.size();
    const double xRatio = std::ceil(mStimuliProvider.getSizeX()
                                    / (double)mOutputsDims[0]);
    const double yRatio = std::ceil(mStimuliProvider.getSizeY()
                                    / (double)mOutputsDims[1]);
    const unsigned int nbLocations = mOutputsDims[1] * mOutputsDims[0];
    const unsigned int miniBatchSize = mLossPositiveSample
                                        + mLossNegativeSample;

    const Tensor<Float_T>& inputsCls = tensor_cast_nocopy<Float_T>(mInputs[0]);
    const Tensor<Float_T>& inputsCoords = (mInputs.size() > 1)
        ? tensor_cast_nocopy<Float_T>(mInputs[1]) : inputsCls;

    Tensor<Float_T> diffOutputsCls
        = tensor_cast_nocopy<Float_T>(mDiffOutputs[0]);
    Tensor<Float_T> diffOutputsCoords = (mDiffOutputs.size() > 1)
        ? tensor_cast_nocopy<Float_T>(mDiffOutputs[1]) : diffOutputsCls;
    const unsigned int coordsOffset = (mDiffOutputs.size() > 1)
        ? 0 : mScoresCls;

#pragma omp parallel for if (mDiffInputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mDiffInputs.dimB(); ++batchPos) {
        std::vector<Tensor<int>::Index> positive;
        std::vector<Tensor<int>::Index> negative;

        for (unsigned int k = 0; k < nbAnchors; ++k) {
            const AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors[k];

            for (unsigned int ya = 0; ya < mOutputsDims[1]; ++ya) {
                for (unsigned int xa = 0; xa < mOutputsDims[0]; ++xa) {
                    // Shifted anchors coordinates at (xa, ya)
                    const int xa0 = (int)(anchor.x0 + xa * xRatio);
                    const int ya0 = (int)(anchor.y0 + ya * yRatio);
                    const int xa1 = (int)(anchor.x1 + xa * xRatio);
                    const int ya1 = (int)(anchor.y1 + ya * yRatio);

                    if (xa0 >= 0
                        && ya0 >= 0
                        && xa1 < (int)mStimuliProvider.getSizeX()
                        && ya1 < (int)mStimuliProvider.getSizeY())
                    {
                        const Float_T IoU = mOutputs(xa, ya,
                                                k + 5 * nbAnchors, batchPos);

                        if (IoU > mPositiveIoU
                            || (mMaxIoU[batchPos] > 0.0
                                && IoU == mMaxIoU[batchPos]))
                        {
                            positive.push_back(
                                Tensor<int>::Index(xa, ya, k, batchPos));
                        }
                        else if (IoU < mNegativeIoU) {
                            negative.push_back(
                                Tensor<int>::Index(xa, ya, k, batchPos));
                        }
                    }

                    diffOutputsCls(xa, ya, k, batchPos) = 0.0;

                    if (mScoresCls > 0)
                        diffOutputsCls(xa, ya, k + nbAnchors, batchPos) = 0.0;

                    diffOutputsCoords(xa, ya,
                        k + coordsOffset * nbAnchors, batchPos) = 0.0;
                    diffOutputsCoords(xa, ya,
                        k + (coordsOffset + 1) * nbAnchors, batchPos) = 0.0;
                    diffOutputsCoords(xa, ya,
                        k + (coordsOffset + 2) * nbAnchors, batchPos) = 0.0;
                    diffOutputsCoords(xa, ya,
                        k + (coordsOffset + 3) * nbAnchors, batchPos) = 0.0;
                }
            }
        }
/*
        // DEBUG
        std::cout << "Found " << positive.size() << " positive IoU\n"
            "Found " << negative.size() << " negative IoU\n"
            "Max IoU = " << mMaxIoU[batchPos] << std::endl;
*/
        for (unsigned int n = 0; n < miniBatchSize; ++n) {
            const bool isPositive = (n < mLossPositiveSample
                                     && !positive.empty());

            std::vector<Tensor<int>::Index>& anchorIndex =
                                    (isPositive)
                                        ? positive
                                        : negative;

            if (anchorIndex.empty()) {
                std::cout << Utils::cwarning << "Warning: not enough negative"
                    " samples!" << Utils::cdef << std::endl;
                break;
            }

            const unsigned int idx = Random::randUniform(0,
                                        anchorIndex.size() - 1);
            const unsigned int xa = anchorIndex[idx][0];
            const unsigned int ya = anchorIndex[idx][1];
            const unsigned int k = anchorIndex[idx][2];

             diffOutputsCls(xa, ya, k, batchPos)
                 = (isPositive - inputsCls(xa, ya, k, batchPos)) / miniBatchSize;

            if (coordsOffset > 0) {
                diffOutputsCoords(xa, ya, k + nbAnchors, batchPos)
                    = ((!isPositive) - inputsCoords(xa, ya, k + nbAnchors, batchPos))
                        / miniBatchSize;
            }

            if (isPositive) {
                const int argMaxIoU = mArgMaxIoU(anchorIndex[idx]);

                // Ground Truth box coordinates
                const AnchorCell_Frame_Kernels::BBox_T& gt
                    = mGT[batchPos][argMaxIoU];

                const AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors[k];

                // Shifted anchors coordinates at (xa, ya)
                const int xa0 = (int)(anchor.x0 + xa * xRatio);
                const int ya0 = (int)(anchor.y0 + ya * yRatio);
                const int xa1 = (int)(anchor.x1 + xa * xRatio);
                const int ya1 = (int)(anchor.y1 + ya * yRatio);

                // Anchors width and height
                const int wa = xa1 - xa0;
                const int ha = ya1 - ya0;

                // Anchor center coordinates (xac, yac)
                const Float_T xac = xa0 + wa / 2.0;
                const Float_T yac = ya0 + ha / 2.0;

                // Ground Truth center coordinates (xgtc, ygtc)
                const Float_T xgtc = gt.x + gt.w / 2.0;
                const Float_T ygtc = gt.y + gt.h / 2.0;

                // Parameterized Ground Truth center coordinates
                const Float_T txgt = ((mFlip) ? -(xgtc - xac) : (xgtc - xac))
                                        / wa;
                const Float_T tygt = ((mFlip) ? -(ygtc - yac) : (ygtc - yac))
                                        / ha;
                const Float_T twgt = std::log(gt.w / wa);
                const Float_T thgt = std::log(gt.h / ha);

                // Parameterized coordinates
                const Float_T tx = inputsCoords(xa, ya,
                    k + coordsOffset * nbAnchors, batchPos);
                const Float_T ty = inputsCoords(xa, ya,
                    k + (coordsOffset + 1) * nbAnchors, batchPos);
                const Float_T tw = inputsCoords(xa, ya,
                    k + (coordsOffset + 2) * nbAnchors, batchPos);
                const Float_T th = inputsCoords(xa, ya,
                    k + (coordsOffset + 3) * nbAnchors, batchPos);

                // Smooth L1 loss
                diffOutputsCoords(xa, ya, k + coordsOffset * nbAnchors,
                                  batchPos)
                   = mLossLambda * smoothL1(txgt, tx) / nbLocations;
                diffOutputsCoords(xa, ya, k + (coordsOffset + 1) * nbAnchors,
                                  batchPos)
                   = mLossLambda * smoothL1(tygt, ty) / nbLocations;
                diffOutputsCoords(xa, ya, k + (coordsOffset + 2) * nbAnchors,
                                  batchPos)
                   = mLossLambda * smoothL1(twgt, tw) / nbLocations;
                diffOutputsCoords(xa, ya, k + (coordsOffset + 3) * nbAnchors,
                                  batchPos)
                   = mLossLambda * smoothL1(thgt, th) / nbLocations;
            }

            anchorIndex.erase(anchorIndex.begin() + idx);
        }
    }

    mDiffOutputs[0] = diffOutputsCls;

    if (mDiffOutputs.size() > 1)
        mDiffOutputs[1] = diffOutputsCoords;

    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

void N2D2::AnchorCell_Frame::update()
{
    // Nothing to update
}
