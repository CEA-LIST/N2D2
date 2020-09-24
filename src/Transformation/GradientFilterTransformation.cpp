/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
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

#include "Transformation/GradientFilterTransformation.hpp"

const char* N2D2::GradientFilterTransformation::Type = "GradientFilter";

N2D2::GradientFilterTransformation::GradientFilterTransformation(
    double scale, double delta, bool applyToLabels)
    : mScale(scale),
      mDelta(delta),
      mApplyToLabels(applyToLabels),
      mGradientFilter(this, "GradientFilter", Sobel),
      mKernelSize(this, "KernelSize", 3),
      mInvThreshold(this, "InvThreshold", false),
      mThreshold(this, "Threshold", 0.5),
      mLabel(this, "Label", std::vector<int>()),
      mGradientScale(this, "GradientScale", 1.0)
{
    // ctor
}

N2D2::GradientFilterTransformation::GradientFilterTransformation(
    const GradientFilterTransformation& trans)
    : mScale(trans.mScale),
      mDelta(trans.mDelta),
      mApplyToLabels(trans.mApplyToLabels),
      mGradientFilter(this, "GradientFilter", trans.mGradientFilter),
      mKernelSize(this, "KernelSize", trans.mKernelSize),
      mInvThreshold(this, "InvThreshold", trans.mInvThreshold),
      mThreshold(this, "Threshold", trans.mThreshold),
      mLabel(this, "Label", trans.mLabel),
      mGradientScale(this, "GradientScale", trans.mGradientScale)
{
    // copy-ctor
}

void N2D2::GradientFilterTransformation::apply(cv::Mat& frame,
                                         cv::Mat& labels,
                                         std::vector
                                         <std::shared_ptr<ROI> >& /*labelsROI*/,
                                         int /*id*/)
{
    double maxValue = 1.0;

    switch (frame.depth()) {
    case CV_8U:
        maxValue = 255;
        break;
    case CV_8S:
        maxValue = 127;
        break;
    case CV_16U:
        maxValue = 65535;
        break;
    case CV_16S:
        maxValue = 32767;
        break;
    case CV_32S:
        maxValue = 2147483647;
        break;
    default:
        break;
    }

    cv::Mat mat;
    frame.convertTo(mat, CV_32F, 1.0 / maxValue);

    if (mApplyToLabels && mGradientScale != 1.0) {
        cv::Mat matScaled;
        cv::resize(mat, matScaled, cv::Size(0,0),
                   mGradientScale, mGradientScale);
        mat = matScaled;
    }

    // Compute derivative
    cv::Mat frameDeriv;

    if (mGradientFilter == Sobel) {
        cv::Mat frameDerivX, frameDerivY;
        cv::Sobel(mat, frameDerivX, CV_32F, 1, 0, mKernelSize, mScale, mDelta);
        cv::Sobel(mat, frameDerivY, CV_32F, 0, 1, mKernelSize, mScale, mDelta);
        cv::addWeighted(cv::abs(frameDerivX), 0.5,
                        cv::abs(frameDerivY), 0.5, 0, frameDeriv);
    }
    else if (mGradientFilter == Scharr) {
        cv::Mat frameDerivX, frameDerivY;
        cv::Scharr(mat, frameDerivX, CV_32F, 1, 0, mScale, mDelta);
        cv::Scharr(mat, frameDerivY, CV_32F, 0, 1, mScale, mDelta);
        cv::addWeighted(cv::abs(frameDerivX), 0.5,
                        cv::abs(frameDerivY), 0.5, 0, frameDeriv);
    }
    else if (mGradientFilter == Laplacian) {
        cv::Laplacian(mat, frameDeriv, CV_32F, mKernelSize, mScale, mDelta);
        frameDeriv = cv::abs(frameDeriv);
    }

    if (mApplyToLabels) {
        // Thresholding
        cv::Mat frameThres;
        cv::threshold(frameDeriv, frameThres, mThreshold, 1.0,
                      (mInvThreshold) ? cv::THRESH_BINARY_INV
                                      : cv::THRESH_BINARY);

        if (mGradientScale != 1.0) {
            cv::Mat frameThresScaled;
            cv::resize(frameThres, frameThresScaled,
                       cv::Size(frame.cols, frame.rows));
            frameThres = frameThresScaled;
        }

        // Label masking
        cv::Mat newLabels = cv::Mat(cv::Size(labels.cols, labels.rows),
                                    CV_32S, cv::Scalar(-1));

        if (mLabel->empty())
            labels.copyTo(newLabels, (frameThres > 0.0));
        else {
            cv::Mat mask = (labels != (*mLabel->begin()));

            for (std::vector<int>::const_iterator it = mLabel->begin() + 1,
                 itEnd = mLabel->end(); it != itEnd; ++it)
            {
                mask = mask & (labels != (*it));
            }

            labels.copyTo(newLabels, (frameThres > 0.0) | mask);
        }

        labels = newLabels;
    }
    else {
        frame = frameDeriv;
    }
}

N2D2::GradientFilterTransformation::~GradientFilterTransformation() {
    
}
