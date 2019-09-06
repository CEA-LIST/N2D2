/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Transformation/PadCropTransformation.hpp"

const char* N2D2::PadCropTransformation::Type = "PadCrop";

N2D2::PadCropTransformation::PadCropTransformation(unsigned int width,
                                                   unsigned int height)
    : mWidth(width),
      mHeight(height),
      mBorderType(this, "BorderType", MinusOneReflectBorder),
      mBorderValue(this, "BorderValue", std::vector<double>())
{
    // ctor
}

N2D2::PadCropTransformation::PadCropTransformation(
    const PadCropTransformation& trans)
    : mWidth(trans.mWidth),
      mHeight(trans.mHeight),
      mBorderType(this, "BorderType", trans.mBorderType),
      mBorderValue(this, "BorderValue", trans.mBorderValue)
{
    // copy-ctor
}

void N2D2::PadCropTransformation::apply(cv::Mat& frame,
                                        cv::Mat& labels,
                                        std::vector
                                        <std::shared_ptr<ROI> >& labelsROI,
                                        int /*id*/)
{
    const int borderType = (mBorderType == MeanBorder)
                                ? cv::BORDER_CONSTANT
                                : mBorderType;

    std::vector<double> bgColorValue = mBorderValue;
    bgColorValue.resize(4, 0.0);
    const cv::Scalar bgColor = (mBorderType == MeanBorder)
        ? cv::mean(frame)
        : cv::Scalar(bgColorValue[0], bgColorValue[1],
                     bgColorValue[2], bgColorValue[3]);

    padCrop(frame, frame.cols, frame.rows, mWidth, mHeight, borderType, bgColor,
        labelsROI);

    if (labels.rows > 1 || labels.cols > 1) {
        std::vector<std::shared_ptr<ROI> > emptyLabelsROI;
        padCrop(labels,
                labels.cols,
                labels.rows,
                mWidth,
                mHeight,
                cv::BORDER_CONSTANT,
                cv::Scalar::all(-1),
                emptyLabelsROI);
    }
}

void N2D2::PadCropTransformation::reverse(cv::Mat& frame,
                                          cv::Mat& labels,
                                          std::vector
                                          <std::shared_ptr<ROI> >& labelsROI,
                                          int /*id*/)
{
    padCrop(labels,
            mWidth,
            mHeight,
            frame.cols,
            frame.rows,
            cv::BORDER_CONSTANT,
            cv::Scalar::all(-1),
            labelsROI);
}

void
N2D2::PadCropTransformation::padCrop(cv::Mat& mat,
                                     unsigned int matWidth,
                                     unsigned int matHeight,
                                     unsigned int width,
                                     unsigned int height,
                                     int borderType,
                                     const cv::Scalar& bgColor,
                                     std::vector
                                     <std::shared_ptr<ROI> >& labelsROI) const
{
    const int dw = width - matWidth;
    const int dh = height - matHeight;

    const int top = std::ceil(dh / 2.0);
    const int bottom = dh - top;
    const int left = std::ceil(dw / 2.0);
    const int right = dw - left;

    if (!mat.empty()) {
        // Padding
        if (dh > 0 || dw > 0) {
            cv::Mat frameBorder;
            cv::copyMakeBorder(mat,
                               frameBorder,
                               std::max(0, top),
                               std::max(0, bottom),
                               std::max(0, left),
                               std::max(0, right),
                               borderType,
                               bgColor);
            mat = frameBorder;
        }

        // Cropping
        if (dh < 0 || dw < 0) {
            const cv::Rect crop(
                cv::Point(std::max(0, -left), std::max(0, -top)),
                cv::Point(mat.cols + std::min(0, right),
                          mat.rows + std::min(0, bottom)));
            mat = mat(crop);
        }
    }

    padCropLabelsROI(labelsROI, -left, -top, width, height);
}
