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

N2D2::AnchorCell_Frame::AnchorCell_Frame(const std::string& name,
                                         StimuliProvider& sp,
                                         const std::vector<Anchor>& anchors)
    : Cell(name, 6*anchors.size()),
      AnchorCell(name, sp, anchors),
      Cell_Frame(name, 6*anchors.size())
{
    // ctor
}

const std::vector<N2D2::AnchorCell::BBox_T>&
N2D2::AnchorCell_Frame::getGT(unsigned int batchPos) const
{
    assert(batchPos < mGT.size());

    return mGT[batchPos];
}

std::shared_ptr<N2D2::ROI>
N2D2::AnchorCell_Frame::getAnchorROI(const Tensor4d<int>::Index& index) const
{
    const int argMaxIoU = mArgMaxIoU(index);

    if (argMaxIoU >= 0) {
        std::vector<std::shared_ptr<ROI> > labelROIs
            = mStimuliProvider.getLabelsROIs(index.b);
        assert(argMaxIoU < (int)labelROIs.size());
        return labelROIs[argMaxIoU];
    }
    else
        return std::shared_ptr<ROI>();
}

N2D2::AnchorCell::BBox_T
N2D2::AnchorCell_Frame::getAnchorBBox(const Tensor4d<int>::Index& index) const
{
    assert(index.i < mArgMaxIoU.dimX());
    assert(index.j < mArgMaxIoU.dimY());
    assert(index.k < mArgMaxIoU.dimZ());
    assert(index.b < mArgMaxIoU.dimB());

    const unsigned int nbAnchors = mAnchors.size();
    const Float_T xa = mOutputs(index.i, index.j,
                                index.k + 1 * nbAnchors, index.b);
    const Float_T ya = mOutputs(index.i, index.j,
                                index.k + 2 * nbAnchors, index.b);
    const Float_T wa = mOutputs(index.i, index.j,
                                index.k + 3 * nbAnchors, index.b);
    const Float_T ha = mOutputs(index.i, index.j,
                                index.k + 4 * nbAnchors, index.b);
    return BBox_T(xa, ya, wa, ha);
}

N2D2::AnchorCell::BBox_T
N2D2::AnchorCell_Frame::getAnchorGT(const Tensor4d<int>::Index& index) const
{
    assert(index.b < mGT.size());

    const int argMaxIoU = mArgMaxIoU(index);

    assert(argMaxIoU < 0 || argMaxIoU < mGT[index.b].size());

    return ((argMaxIoU >= 0)
        ? mGT[index.b][argMaxIoU]
        : BBox_T(0.0, 0.0, 0.0, 0.0));
}

N2D2::Float_T
N2D2::AnchorCell_Frame::getAnchorIoU(const Tensor4d<int>::Index& index) const
{
    assert(index.i < mArgMaxIoU.dimX());
    assert(index.j < mArgMaxIoU.dimY());
    assert(index.k < mArgMaxIoU.dimZ());
    assert(index.b < mArgMaxIoU.dimB());

    return mOutputs(index.i, index.j, index.k + 5 * mAnchors.size(), index.b);
}

int
N2D2::AnchorCell_Frame::getAnchorArgMaxIoU(const Tensor4d<int>::Index& index)
    const
{
    return mArgMaxIoU(index);
}

void N2D2::AnchorCell_Frame::initialize()
{
    const unsigned int nbAnchors = mAnchors.size();

    if (mInputs.dimZ() != 5 * nbAnchors) {
        throw std::domain_error("AnchorCell_Frame::initialize():"
                                " the number of input channels must be equal to"
                                " 5 times the number of anchors.");
    }

    if (mInputs.size() > 1 && mInputs[0].dimZ() != nbAnchors) {
        throw std::domain_error("AnchorCell_Frame::initialize():"
                                " the first input number of channels must match"
                                " the number of anchors.");
    }

    mGT.resize(mOutputs.dimB());
    mArgMaxIoU.resize(mOutputsWidth,
                      mOutputsHeight,
                      mAnchors.size(),
                      mOutputs.dimB());
}

void N2D2::AnchorCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDToH();

    const unsigned int nbAnchors = mAnchors.size();
    const double xRatio = mStimuliProvider.getSizeX() / (double)mOutputs.dimX();
    const double yRatio = mStimuliProvider.getSizeY() / (double)mOutputs.dimY();

#pragma omp parallel for if (mOutputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
        std::vector<BBox_T>& GT = mGT[batchPos];

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

            GT[l] = std::make_tuple(labelRect.tl().x,
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
            const std::vector<BBox_T>& GT = mGT[batchPos];
            const Anchor& anchor = mAnchors[k];
            const Float_T wa = anchor.width;
            const Float_T ha = anchor.height;
/*
            // DEBUG
            std::cout << "Number of GT boxes: " << GT.size() << std::endl;

            for (unsigned int i = 0, size = GT.size(); i < size; ++i) {
                int xgt, ygt, wgt, hgt;
                std::tie(xgt, ygt, wgt, hgt) = GT[i];

                std::cout << "  " << wgt << "x" << hgt
                    << " @ (" << xgt << "," << ygt << ")" << std::endl;
            }

            cv::Mat imgIoUHsv(cv::Size(mOutputsWidth, mOutputsHeight),
                                    CV_8UC3,
                                    cv::Scalar(0, 0, 0));
            cv::Mat imgClsHsv(cv::Size(mOutputsWidth, mOutputsHeight),
                                    CV_8UC3,
                                    cv::Scalar(0, 0, 0));
*/
            for (unsigned int ya = 0; ya < mOutputsHeight; ++ya) {
                for (unsigned int xa = 0; xa < mOutputsWidth; ++xa) {
                    const Float_T xac = xa * xRatio - wa / 2.0;
                    const Float_T yac = ya * yRatio - ha / 2.0;

                    /**
                     * 1st condition: "During  training,  we  ignore all
                     * cross-boundary anchors so they do not contribute to  the
                     * loss."
                     * 2nd condition: "During testing, however, we still apply
                     * the fully convolutional RPN  to  the  entire  image."
                    */
                    if ((xac >= 0.0
                        && yac >= 0.0
                        && xac + wa < mStimuliProvider.getSizeX()
                        && yac + ha < mStimuliProvider.getSizeY())
                        || inference)
                    {
                        // Score
                        const Float_T cls = mInputs(xa, ya, k, batchPos);

                        // Parameterized coordinates
                        const Float_T txbb = mInputs(xa, ya,
                                                   k + 1 * nbAnchors, batchPos);
                        const Float_T tybb = mInputs(xa, ya,
                                                   k + 2 * nbAnchors, batchPos);
                        const Float_T twbb = mInputs(xa, ya,
                                                   k + 3 * nbAnchors, batchPos);
                        const Float_T thbb = mInputs(xa, ya,
                                                   k + 4 * nbAnchors, batchPos);

                        // Predicted box coordinates
                        Float_T xbb = txbb * wa + xac;
                        Float_T ybb = tybb * ha + yac;
                        Float_T wbb = wa * std::exp(twbb);
                        Float_T hbb = ha * std::exp(thbb);

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
                            if (xbb + wbb > mStimuliProvider.getSizeX())
                                wbb = mStimuliProvider.getSizeX() - xbb;
                            if (ybb + hbb > mStimuliProvider.getSizeY())
                                hbb = mStimuliProvider.getSizeY() - ybb;
                        }

                        // For inference, compute IoU on predicted boxes
                        // For learning, compute IoU on anchor boxes
                        // => if IoU is computed on predicted boxes during
                        // learning, predicted boxes may arbitrarily drift from
                        // anchors and learning does not converge
                        Float_T x, y, w, h;
                        std::tie(x, y, w, h)
                            = (inference) ? BBox_T(xbb, ybb, wbb, hbb)
                                          : BBox_T(xac, yac, wa, ha);

                        Float_T maxIoU = 0.0;
                        int argMaxIoU = -1;

                        for (unsigned int l = 0, nbLabels = GT.size();
                            l < nbLabels; ++l)
                        {
                            // Ground Truth box coordinates
                            Float_T xgt, ygt, wgt, hgt;
                            std::tie(xgt, ygt, wgt, hgt) = GT[l];

                            const Float_T interLeft = std::max(xgt, x);
                            const Float_T interRight = std::min(xgt + wgt,
                                                                x + w);
                            const Float_T interTop = std::max(ygt, y);
                            const Float_T interBottom = std::min(ygt + hgt,
                                                                 y + h);

                            if (interLeft < interRight
                                && interTop < interBottom)
                            {
                                const Float_T interArea
                                    = (interRight - interLeft)
                                        * (interBottom - interTop);
                                const Float_T unionArea = wgt * hgt + w * h
                                                        - interArea;
                                const Float_T IoU = interArea / unionArea;

                                if (IoU > maxIoU) {
                                    maxIoU = IoU;
                                    argMaxIoU = l;
                                }
                            }
                        }

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

                for (unsigned int ya = 0; ya < mOutputsHeight; ++ya) {
                    for (unsigned int xa = 0; xa < mOutputsWidth; ++xa) {
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
    const double xRatio = mStimuliProvider.getSizeX() / (double)mOutputs.dimX();
    const double yRatio = mStimuliProvider.getSizeY() / (double)mOutputs.dimY();
    const unsigned int nbLocations = mOutputsHeight * mOutputsWidth;
    const unsigned int miniBatchSize = mLossPositiveSample
                                        + mLossNegativeSample;

#pragma omp parallel for if (mDiffInputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mDiffInputs.dimB(); ++batchPos) {
        std::vector<Tensor4d<int>::Index> positive;
        std::vector<Tensor4d<int>::Index> negative;

        for (unsigned int k = 0; k < nbAnchors; ++k) {
            const Anchor& anchor = mAnchors[k];
            const Float_T wa = anchor.width;
            const Float_T ha = anchor.height;

            for (unsigned int ya = 0; ya < mOutputsHeight; ++ya) {
                for (unsigned int xa = 0; xa < mOutputsWidth; ++xa) {
                    const Float_T xac = xa * xRatio - wa / 2.0;
                    const Float_T yac = ya * yRatio - ha / 2.0;

                    if (xac >= 0.0
                        && yac >= 0.0
                        && xac + wa < mStimuliProvider.getSizeX()
                        && yac + ha < mStimuliProvider.getSizeY())
                    {
                        const Float_T IoU = mOutputs(xa, ya,
                                                k + 5 * nbAnchors, batchPos);

                        if (IoU > mPositiveIoU
                            || (mMaxIoU[batchPos] > 0.0
                                && IoU == mMaxIoU[batchPos]))
                        {
                            positive.push_back(
                                Tensor4d<int>::Index(xa, ya, k, batchPos));
                        }
                        else if (IoU < mNegativeIoU) {
                            negative.push_back(
                                Tensor4d<int>::Index(xa, ya, k, batchPos));
                        }
                    }

                    mDiffOutputs(xa, ya, k, batchPos) = 0.0;
                    mDiffOutputs(xa, ya, k + 1 * nbAnchors, batchPos) = 0.0;
                    mDiffOutputs(xa, ya, k + 2 * nbAnchors, batchPos) = 0.0;
                    mDiffOutputs(xa, ya, k + 3 * nbAnchors, batchPos) = 0.0;
                    mDiffOutputs(xa, ya, k + 4 * nbAnchors, batchPos) = 0.0;
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

            std::vector<Tensor4d<int>::Index>& anchorIndex =
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
            const unsigned int xa = anchorIndex[idx].i;
            const unsigned int ya = anchorIndex[idx].j;
            const unsigned int k = anchorIndex[idx].k;

            if (isPositive) {
                mDiffOutputs(xa, ya, k, batchPos)
                    = (1.0 - mInputs(xa, ya, k, batchPos)) / miniBatchSize;

                const int argMaxIoU = mArgMaxIoU(anchorIndex[idx]);

                // Ground Truth box coordinates
                int xgt, ygt, wgt, hgt;
                std::tie(xgt, ygt, wgt, hgt) = mGT[batchPos][argMaxIoU];

                // Parameterized Ground Truth coordinates
                const Anchor& anchor = mAnchors[k];
                const Float_T wa = anchor.width;
                const Float_T ha = anchor.height;

                const Float_T txgt = (xgt - xa * xRatio + wa / 2.0) / wa;
                const Float_T tygt = (ygt - ya * yRatio + ha / 2.0) / ha;
                const Float_T twgt = std::log(wgt / wa);
                const Float_T thgt = std::log(hgt / ha);

                // Parameterized coordinates
                const Float_T tx = mInputs(xa, ya, k + 1 * nbAnchors, batchPos);
                const Float_T ty = mInputs(xa, ya, k + 2 * nbAnchors, batchPos);
                const Float_T tw = mInputs(xa, ya, k + 3 * nbAnchors, batchPos);
                const Float_T th = mInputs(xa, ya, k + 4 * nbAnchors, batchPos);

                // Smooth L1 loss
                mDiffOutputs(xa, ya, k + 1 * nbAnchors, batchPos)
                   = mLossLambda * smoothL1(txgt, tx) / nbLocations;
                mDiffOutputs(xa, ya, k + 2 * nbAnchors, batchPos)
                   = mLossLambda * smoothL1(tygt, ty) / nbLocations;
                mDiffOutputs(xa, ya, k + 3 * nbAnchors, batchPos)
                   = mLossLambda * smoothL1(twgt, tw) / nbLocations;
                mDiffOutputs(xa, ya, k + 4 * nbAnchors, batchPos)
                   = mLossLambda * smoothL1(thgt, th) / nbLocations;
            }
            else {
                mDiffOutputs(xa, ya, k, batchPos)
                    = (0.0 - mInputs(xa, ya, k, batchPos)) / miniBatchSize;
            }

            anchorIndex.erase(anchorIndex.begin() + idx);
        }
    }

    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

void N2D2::AnchorCell_Frame::update()
{
    // Nothing to update
}
