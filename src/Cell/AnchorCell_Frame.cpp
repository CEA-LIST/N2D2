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

#include "ROI/ROI.hpp"
#include "Cell/AnchorCell_Frame.hpp"
#include "DeepNet.hpp"
#include "StimuliProvider.hpp"

N2D2::Registrar<N2D2::AnchorCell>
N2D2::AnchorCell_Frame::mRegistrar("Frame", N2D2::AnchorCell_Frame::create);

N2D2::AnchorCell_Frame::AnchorCell_Frame(
    const DeepNet& deepNet,
    const std::string& name,
    StimuliProvider& sp,
    const AnchorCell_Frame_Kernels::DetectorType detectorType,
    const AnchorCell_Frame_Kernels::Format inputFormat,
    const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
    unsigned int scoresCls)
    : Cell(deepNet, name, 6*anchors.size()),
      AnchorCell(deepNet, name, sp, detectorType, inputFormat, anchors, scoresCls),
      Cell_Frame<Float_T>(deepNet, name, 6*anchors.size()),
      mAnchors(anchors)
{
    // ctor
}
int N2D2::AnchorCell_Frame::getNbAnchors() const { return mAnchors.size(); }

void N2D2::AnchorCell_Frame::setAnchors(const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors)
{
    //Reload new anchors generated from KMeans clustering

    mAnchors.resize(anchors.size());
    for(unsigned int i = 0; i < anchors.size(); ++i)
        mAnchors[i] = anchors[i];

    std::cout << "N2D2::AnchorCell_Frame::setAnchors" << 
                "Reload new anchors generated from KMeans clustering" << std::endl;
}

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
    if (mNbClass < 1) {
        throw std::domain_error("AnchorCell_Frame::initialize():"
                                " the number of classes must be superior to 0");
    }

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
    if(mDetectorType == AnchorCell_Frame_Kernels::DetectorType::LapNet)
    {
        mGTClass.resize(mOutputs.dimB());
        for(unsigned int b = 0; b <mOutputs.dimB(); ++b )
            mGTClass[b].resize(mNbClass);
    }
}

void N2D2::AnchorCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

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
                                    / (float) (mFeatureMapWidth);
    const float yOutputRatio = mStimuliProvider.getSizeY()
                                    / (float) (mFeatureMapHeight);
    if(mDetectorType == AnchorCell_Frame_Kernels::DetectorType::LapNet)
    {

#pragma omp parallel for if (mOutputs.dimB() > 4)
        for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
            std::vector<std::vector<AnchorCell_Frame_Kernels::BBox_T> >& GT = mGTClass[batchPos];
            //GT.resize(mNbClass);

            // Extract ground true ROIs
            std::vector<std::shared_ptr<ROI> > labelROIs
                = mStimuliProvider.getLabelsROIs(batchPos);
            for( int c = 0; c < mNbClass; ++c)
                GT[c].resize(0);

            if(labelROIs.size() > mMaxLabelGT)
            {
                std::cout << Utils::cwarning << "AnchorCell_Frame::propagate(): Number of ground truth labels "
                    "(" << labelROIs.size() << ") is superior of the MaxLabelPerFrame defined as "  << "("
                    << mMaxLabelGT << ")" << Utils::cdef << std::endl;
            }

            for (unsigned int l = 0; l < labelROIs.size(); ++l) {
                cv::Rect labelRect = labelROIs[l]->getBoundingRect();
                //std::cout << " " <<  mStimuliProvider.getDatabase().getLabelName(labelROIs[l]->getLabel())
                //            << "->" << mLabelsMapping[labelROIs[l]->getLabel()];
                int cls = mLabelsMapping[labelROIs[l]->getLabel()];

                if(cls > -1)
                {

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

                    GT[cls].push_back(AnchorCell_Frame_Kernels::BBox_T( labelRect.tl().x,
                                                                        labelRect.tl().y,
                                                                        labelRect.width, labelRect.height));
                }
            }
            //std::cout << std::endl;
        }


        mMaxIoUClass.resize(mOutputs.dimB());
        for(unsigned int b = 0; b < mOutputs.dimB(); ++b)
            mMaxIoUClass[b].assign(mNbClass, 0.0);
    //    const unsigned int size = mOutputs.dimB() * nbAnchors;
    //#if defined(_OPENMP) && _OPENMP >= 200805
    //#pragma omp parallel for collapse(2) if (size > 16)
    //#else
    //#pragma omp parallel for if (mOutputs.dimB() > 4 && size > 16)
    //#endif

        for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
               /* // DEBUG
                std::cout << "Number of GT boxes: " << GT[classIdx].size() << std::endl;

                for (unsigned int i = 0, size = GT[classIdx].size(); i < size; ++i) {
                    //int xgt, ygt, wgt, hgt;
                    //std::tie(xgt, ygt, wgt, hgt) = GT[i];
                    int xgt = GT[classIdx][i].x;
                    int ygt = GT[classIdx][i].y;
                    int wgt = GT[classIdx][i].w;
                    int hgt = GT[classIdx][i].h;

                    std::cout << "  " << wgt << "x" << hgt
                        << " @ (" << xgt << "," << ygt << ")" << std::endl;
                }

                cv::Mat imgIoUHsv(cv::Size(mOutputsDims[0], mOutputsDims[1]),
                                        CV_8UC3,
                                        cv::Scalar(0, 0, 0));
                cv::Mat imgClsHsv(cv::Size(mOutputsDims[0], mOutputsDims[1]),
                                        CV_8UC3,
                                        cv::Scalar(0, 0, 0));
                cv::Mat imgCls;*/

            for (unsigned int k = 0; k < nbAnchors; ++k) {
                const std::vector<std::vector<AnchorCell_Frame_Kernels::BBox_T> >& GT
                    = mGTClass[batchPos];
                const AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors[k];
                const int classIdx = k/(nbAnchors/mNbClass);


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
                        /*if ((xa0 >= 0
                            && ya0 >= 0
                            && xa1 < (int)mFeatureMapWidth
                            && ya1 < (int)mFeatureMapHeight)
                            || inference)
                        {*/
                            // Score
                            const Float_T cls = inputsCls(xa, ya, k, batchPos);

                            // Parameterized coordinates
                            const std::size_t txOffset = (mInputFormat == AnchorCell_Frame_Kernels::Format::CA) ?
                                                        k + coordsOffset * nbAnchors :
                                                        k*4;
                            const std::size_t tyOffset = (mInputFormat == AnchorCell_Frame_Kernels::Format::CA) ?
                                                        k + (coordsOffset + 1) * nbAnchors :
                                                        k*4 + 1;
                            const std::size_t twOffset = (mInputFormat == AnchorCell_Frame_Kernels::Format::CA) ?
                                                        k + (coordsOffset + 2) * nbAnchors :
                                                        k*4 + 2;
                            const std::size_t thOffset = (mInputFormat == AnchorCell_Frame_Kernels::Format::CA) ?
                                                        k + (coordsOffset + 3) * nbAnchors :
                                                        k*4 + 3;


                            const Float_T txbb = std::max(std::min(inputsCoords(xa, ya,
                                                                                txOffset, batchPos), 70.0f), -70.0f);

                            const Float_T tybb = std::max(std::min(inputsCoords(xa, ya,
                                                                                tyOffset, batchPos), 70.0f), -70.0f);

                            const Float_T twbb = std::max(std::min(inputsCoords(xa, ya,
                                                                                twOffset, batchPos), 70.0f), -70.0f);

                            const Float_T thbb = std::max(std::min(inputsCoords(xa, ya,
                                                                                thOffset, batchPos), 70.0f), -70.0f);

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

                            //if (!inference) {
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
                            //}

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
                            //std::cout << "Class: " << k/(nbAnchors/mNbClass);

                            for (unsigned int l = 0, nbLabels = GT[classIdx].size();
                                l < nbLabels; ++l)
                            {
                                // Ground Truth box coordinates
                                const AnchorCell_Frame_Kernels::BBox_T& gt = GT[classIdx][l];

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
                            mMaxIoUClass[batchPos][classIdx] = std::max(mMaxIoUClass[batchPos][classIdx], maxIoU);
/*
                        //}
                        else {
                            mOutputs(xa, ya, k, batchPos) = 0.0;
                            mOutputs(xa, ya, k + 1 * nbAnchors, batchPos) = 0.0;
                            mOutputs(xa, ya, k + 2 * nbAnchors, batchPos) = 0.0;
                            mOutputs(xa, ya, k + 3 * nbAnchors, batchPos) = 0.0;
                            mOutputs(xa, ya, k + 4 * nbAnchors, batchPos) = 0.0;
                            mOutputs(xa, ya, k + 5 * nbAnchors, batchPos) = 0.0;
                            mArgMaxIoU(xa, ya, k, batchPos) = -1;
                        }*/
                    }
                }


            }
/*
            const double alpha = 0.25;
            Utils::createDirectories(Utils::filePath(mName));

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
            cv::cvtColor(img8U, imgColor, cv::COLOR_GRAY2BGR);
            cv::Mat imgBlended;
            std::string fileName = Utils::filePath(mName) + "/anchors.png";

            for (unsigned int k = 0; k < nbAnchors; ++k) {
                const std::vector<std::vector<AnchorCell_Frame_Kernels::BBox_T> >& GT
                    = mGTClass[batchPos];
                const AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors[k];
                const int classIdx = k/(nbAnchors/mNbClass);

                // DEBUG
                if (batchPos == 0) {
                    for (unsigned int l = 0, nbLabels = GT[classIdx].size(); l < nbLabels;
                        ++l)
                    {

                        //Float_T xgt, ygt, wgt, hgt;
                        //std::tie(xgt, ygt, wgt, hgt) = GT[l];
                        Float_T xgt = GT[classIdx][l].x;
                        Float_T ygt = GT[classIdx][l].y;
                        Float_T wgt = GT[classIdx][l].w;
                        Float_T hgt = GT[classIdx][l].h;
                        double max = 0.0;
                        float final_x0 = 0.0;
                        float final_y0 = 0.0;
                        float final_x1 = 0.0;
                        float final_y1 = 0.0;

                        for (unsigned int ya = 0; ya < mOutputsDims[1]; ++ya) {
                            for (unsigned int xa = 0; xa < mOutputsDims[0]; ++xa) {
                                const Float_T xa0 = (anchor.x0 + xa * xRatio);
                                const Float_T ya0 = (anchor.y0 + ya * yRatio);
                                const Float_T xa1 = (anchor.x1 + xa * xRatio);
                                const Float_T ya1 = (anchor.y1 + ya * yRatio);

                                const Float_T interLeft = std::max(xgt, std::max(xa0, 0.0f));
                                const Float_T interRight = std::min(xgt + wgt, std::max(xa1, 0.0f));
                                const Float_T interTop = std::max(ygt, std::max(ya0, 0.0f));
                                const Float_T interBottom = std::min(ygt + hgt, std::max(ya1, 0.0f));
                                if (interLeft < interRight
                                    && interTop < interBottom)
                                {
                                    const Float_T interArea
                                        = (interRight - interLeft)
                                            * (interBottom - interTop);
                                    const Float_T unionArea = wgt * hgt
                                        + (xa1 - xa0) * (ya1 - ya0) - interArea;
                                    const Float_T IoU = interArea / unionArea;

                                    if (IoU > mPositiveIoU && IoU > max) {
                                        final_x0 = xa0;
                                        final_x1 = xa1;
                                        final_y0 = ya0;
                                        final_y1 = ya1;
                                        max = IoU;

                                    }
                                }

                            }
                        }
                        cv::rectangle(imgColor, cv::Point((int)final_x0, (int)final_y0),
                                                cv::Point((int)final_x1, (int)final_y1),
                                                cv::Scalar(0, 255, 0));

                        cv::rectangle(imgColor,
                                    cv::Point((int)xgt, (int)ygt),
                                    cv::Point((int)(xgt + wgt), (int)(ygt + hgt)),
                                    cv::Scalar(255, 0, 0));

                    }
                    // Target image IoU
                    cv::Mat imgIoU;
                    cv::cvtColor(imgIoUHsv, imgIoU, cv::COLOR_HSV2BGR);

                    cv::Mat imgSampled;
                    cv::resize(imgIoU,
                            imgSampled,
                            cv::Size(mStimuliProvider.getSizeX(),
                                        mStimuliProvider.getSizeY()),
                            0.0,
                            0.0,
                            cv::INTER_NEAREST);

                    cv::addWeighted(
                        imgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);


                    if (!cv::imwrite(fileName, imgBlended))
                        throw std::runtime_error("Unable to write image: "
                                                + fileName);

                    // Target image Cls
                    cv::cvtColor(imgClsHsv, imgCls, cv::COLOR_HSV2BGR);

                    cv::resize(imgCls,
                            imgSampled,
                            cv::Size(mStimuliProvider.getSizeX(),
                                        mStimuliProvider.getSizeY()),
                            0.0,
                            0.0,
                            cv::INTER_NEAREST);

                    cv::addWeighted(
                        imgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);

                    fileName = Utils::filePath(mName) + "/anchors_cls.png";

                    if (!cv::imwrite(fileName, imgBlended))
                        throw std::runtime_error("Unable to write image: "
                                                + fileName);
                }
            }
            */
        }
    }
    else
    {
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


        mMaxIoU.assign(mOutputs.dimB(), 0.0);

        for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
            for (unsigned int k = 0; k < nbAnchors; ++k) {
                const std::vector<AnchorCell_Frame_Kernels::BBox_T>& GT
                    = mGT[batchPos];
                const AnchorCell_Frame_Kernels::Anchor& anchor = mAnchors[k];

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
            }
        }
    }

    Cell_Frame<Float_T>::propagate(inference);
    mDiffInputs.clearValid();
}

void N2D2::AnchorCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    Cell_Frame<Float_T>::backPropagate();

    const unsigned int nbAnchors = mAnchors.size();
    const double xRatio = std::ceil(mStimuliProvider.getSizeX()
                                    / (double)mOutputsDims[0]);
    const double yRatio = std::ceil(mStimuliProvider.getSizeY()
                                    / (double)mOutputsDims[1]);

    const Tensor<Float_T>& inputsCls = tensor_cast_nocopy<Float_T>(mInputs[0]);
    const Tensor<Float_T>& inputsCoords = (mInputs.size() > 1)
        ? tensor_cast_nocopy<Float_T>(mInputs[1]) : inputsCls;

    Tensor<Float_T> diffOutputsCls
        = tensor_cast_nocopy<Float_T>(mDiffOutputs[0]);
    Tensor<Float_T> diffOutputsCoords = (mDiffOutputs.size() > 1)
        ? tensor_cast_nocopy<Float_T>(mDiffOutputs[1]) : diffOutputsCls;
    const unsigned int coordsOffset = (mDiffOutputs.size() > 1)
        ? 0 : mScoresCls;
    const unsigned int nbLocations = mOutputsDims[1] * mOutputsDims[0];

    if(mDetectorType == AnchorCell_Frame_Kernels::DetectorType::SSD)
    {
        const unsigned int miniBatchSize = mLossPositiveSample
                                            + mLossNegativeSample;

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
    }
    else if(mDetectorType == AnchorCell_Frame_Kernels::DetectorType::LapNet)
    {
        std::vector < std::vector < Float_T > > AvgIOU;
        std::vector < std::vector < Float_T > > AvgConf;

        //AvgIOU.resize(mNbClass, std::vector<Float_T>(mDiffInputs.dimB(), 0.0f) );
        //AvgConf.resize(mNbClass, std::vector<Float_T>(mDiffInputs.dimB(), 0.0f) );


#pragma omp parallel for if (mDiffInputs.dimB() > 4)
        for (int batchPos = 0; batchPos < (int)mDiffInputs.dimB(); ++batchPos) {
            std::vector<std::vector<Tensor<int>::Index> > positive(mNbClass,
                                                                    std::vector<Tensor<int>::Index>(0));
            std::vector< std::vector<std::pair< Tensor<int>::Index, Float_T> > > negative(mNbClass,
                                                                                        std::vector<std::pair< Tensor<int>::Index,
                                                                                        Float_T>>(0));
            positive.resize(mNbClass);
            negative.resize(mNbClass);

            for (unsigned int k = 0; k < nbAnchors; ++k) {

                const int classIdx = k/(nbAnchors/mNbClass);

                for (unsigned int ya = 0; ya < mOutputsDims[1]; ++ya) {
                    for (unsigned int xa = 0; xa < mOutputsDims[0]; ++xa) {

                        const Float_T IoU = mOutputs(xa, ya, k + 5 * nbAnchors, batchPos);
                        const Float_T conf = inputsCls(xa, ya, k, batchPos);

                        if (IoU >= mPositiveIoU && (mArgMaxIoU(xa, ya, k, batchPos) > -1))
                        {
                            positive[classIdx].push_back(Tensor<int>::Index(xa, ya, k, batchPos));
                            //AvgIOU[classIdx][batchPos] += IoU;
                            //AvgConf[classIdx][batchPos] += conf;
                        }
                        else if(mArgMaxIoU(xa, ya, k, batchPos) == -1)
                        {
                            negative[classIdx].push_back(std::make_pair(Tensor<int>::Index(xa, ya, k, batchPos), conf));
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

            if(mAnchorsStats.empty())
                mAnchorsStats.resize(mNbClass, 0);
            for(int cls = 0; cls < mNbClass; ++cls)
            {
                const int nbNegative = (negative[cls].size() > positive[cls].size()*mNegativeRatioSSD) ?
                                        positive[cls].size()*mNegativeRatioSSD
                                        : negative[cls].size();

                const int nbPositive = positive[cls].size();
                std::cout << "[" << cls << "] " << nbNegative << "/"
                          << nbPositive << std::endl;

                std::partial_sort(negative[cls].begin(),
                                negative[cls].begin() + nbNegative,
                                negative[cls].end(),
                                Utils::PairSecondPred<Tensor<int>::Index,
                                Float_T, std::greater<Float_T> >());

                /*std::sort(negative[cls].begin(), negative[cls].end(),
                    Utils::PairSecondPred<Tensor<int>::Index,
                        Float_T, std::greater<Float_T> >());

                std::reverse(negative[cls].begin(), negative[cls].end());*/

                for(int neg = 0; neg < nbNegative; ++ neg)
                {
                    Tensor<int>::Index& anchorIndex = negative[cls][neg].first;

                    const unsigned int xa = anchorIndex[0];
                    const unsigned int ya = anchorIndex[1];
                    const unsigned int k = anchorIndex[2];

                    diffOutputsCls(xa, ya, k, batchPos) = -inputsCls(xa, ya, k, batchPos) 
                                                            / (nbPositive );
                }

                for(int pos = 0; pos < nbPositive; ++ pos)
                {
                    Tensor<int>::Index& anchorIndex = positive[cls][pos];

                    const unsigned int xa = anchorIndex[0];
                    const unsigned int ya = anchorIndex[1];
                    const unsigned int k = anchorIndex[2];

                    const int argMaxIoU = mArgMaxIoU(anchorIndex);
                    // Ground Truth box coordinates
                    const AnchorCell_Frame_Kernels::BBox_T& gt
                        = mGTClass[batchPos][cls][argMaxIoU];

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
                    const Float_T lossTx = mLossLambda * smoothL1(txgt, tx) 
                                            / (nbPositive);
                    const Float_T lossTy = mLossLambda * smoothL1(tygt, ty) 
                                            / (nbPositive);
                    const Float_T lossTw = mLossLambda * smoothL1(twgt, tw) 
                                            / (nbPositive);
                    const Float_T lossTh = mLossLambda * smoothL1(thgt, th) 
                                            / (nbPositive);

                    diffOutputsCoords(xa, ya, k + coordsOffset * nbAnchors, batchPos) = lossTx;
                    diffOutputsCoords(xa, ya, k + (coordsOffset + 1) * nbAnchors, batchPos) = lossTy;
                    diffOutputsCoords(xa, ya, k + (coordsOffset + 2) * nbAnchors, batchPos) = lossTw;
                    diffOutputsCoords(xa, ya, k + (coordsOffset + 3) * nbAnchors, batchPos) = lossTh;
                    diffOutputsCls(xa, ya, k, batchPos) = (1.0f - inputsCls(xa, ya, k, batchPos)) / (nbPositive ); 
                                                          //-(std::max(std::log(1.0f - inputsCls(xa, ya, k, batchPos)), 0.0f))  

                }
/*
                if(nbPositive > 0)
                {
                    AvgIOU[cls][batchPos] /= nbPositive;
                    AvgConf[cls][batchPos] /= nbPositive;
                }*/

                mAnchorsStats[cls] += nbPositive;
            }
        }

        /*for(unsigned int cls = 0; cls < mNbClass; ++cls)
        {
            Float_T avgIoUPerCls = 0.0f;
            Float_T avgConfPerCls = 0.0f;
            //avgIoUPerCls = AvgIOU[cls][0];
            //avgConfPerCls = AvgConf[cls][0];

            avgIoUPerCls = std::accumulate(AvgIOU[cls].begin(), AvgIOU[cls].end(), 0.0f);
            avgConfPerCls = std::accumulate(AvgConf[cls].begin(), AvgConf[cls].end(), 0.0f);

            avgIoUPerCls /= mDiffInputs.dimB();
            avgConfPerCls /= mDiffInputs.dimB();
            
            std::cout << "AvgIoU[" << cls << "]: " << avgIoUPerCls << " AvgConf[" << cls << "]: " << avgConfPerCls << std::endl;
        }*/
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