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

#include "Transformation/RescaleTransformation.hpp"

N2D2::RescaleTransformation::RescaleTransformation(unsigned int width,
                                                   unsigned int height)
    : mWidth(width),
      mHeight(height),
      mKeepAspectRatio(this, "KeepAspectRatio", false),
      mResizeToFit(this, "ResizeToFit", true)
{
    // ctor
}

void N2D2::RescaleTransformation::apply(cv::Mat& frame,
                                        cv::Mat& labels,
                                        std::vector
                                        <std::shared_ptr<ROI> >& labelsROI,
                                        int /*id*/)
{
    resize(frame, cv::INTER_LINEAR, labelsROI);

    if (labels.rows > 1 || labels.cols > 1) {
        std::vector<std::shared_ptr<ROI> > emptyLabelsROI;
        resize(labels, cv::INTER_NEAREST, emptyLabelsROI);
    }
}

void N2D2::RescaleTransformation::reverse(cv::Mat& frame,
                                          cv::Mat& labels,
                                          std::vector
                                          <std::shared_ptr<ROI> >& labelsROI,
                                          int /*id*/)
{
    if (!labels.empty()) {
        cv::Mat labelsResized;
        cv::resize(labels,
                   labelsResized,
                   cv::Size(frame.cols, frame.rows),
                   0,
                   0,
                   cv::INTER_NEAREST);
        labels = labelsResized;
    }

    double xRatio = frame.cols / (double)mWidth;
    double yRatio = frame.rows / (double)mHeight;

    if (mKeepAspectRatio) {
        const double ratio = (mResizeToFit) ? std::max(xRatio, yRatio)
                                            : std::min(xRatio, yRatio);
        xRatio = yRatio = ratio;
    }

    std::for_each(
        labelsROI.begin(),
        labelsROI.end(),
        std::bind(&ROI::rescale, std::placeholders::_1, xRatio, yRatio));
}

void
N2D2::RescaleTransformation::resize(cv::Mat& mat,
                                    int interpolation,
                                    std::vector
                                    <std::shared_ptr<ROI> >& labelsROI) const
{
    cv::Mat matResized;

    double xRatio = mWidth / (double)mat.cols;
    double yRatio = mHeight / (double)mat.rows;

    if (mKeepAspectRatio) {
        const double ratio = (mResizeToFit) ? std::min(xRatio, yRatio)
                                            : std::max(xRatio, yRatio);
        xRatio = yRatio = ratio;

        if (ratio * mat.cols >= 1.0 && ratio * mat.rows >= 1.0)
            cv::resize(mat,
                       matResized,
                       cv::Size(ratio * mat.cols, ratio * mat.rows),
                       0,
                       0,
                       interpolation);
    } else
        cv::resize(
            mat, matResized, cv::Size(mWidth, mHeight), 0, 0, interpolation);

    std::for_each(
        labelsROI.begin(),
        labelsROI.end(),
        std::bind(&ROI::rescale, std::placeholders::_1, xRatio, yRatio));

    mat = matResized;
}
