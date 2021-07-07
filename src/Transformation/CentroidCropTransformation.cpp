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

#include "Transformation/CentroidCropTransformation.hpp"

const char* N2D2::CentroidCropTransformation::Type = "CentroidCrop";

N2D2::CentroidCropTransformation::CentroidCropTransformation(int axis)
    : mAxis(axis)
{
    // ctor
}

void N2D2::CentroidCropTransformation::apply(cv::Mat& frame,
                                        cv::Mat& labels,
                                        std::vector
                                        <std::shared_ptr<ROI> >& /*labelsROI*/,
                                        int /*id*/)
{
    const cv::Moments moments = cv::moments(frame);
    const cv::Point centroid(moments.m10 / moments.m00,
                             moments.m01 / moments.m00);

    const int halfWidth = std::min(centroid.x, frame.cols - centroid.x);
    const int halfHeight = std::min(centroid.y, frame.rows - centroid.y);

    cv::Rect centroidSlice;

    if (mAxis == -1) {
        // Both axis
        centroidSlice = cv::Rect(centroid.x - halfWidth,
                                 centroid.y - halfHeight,
                                 2 * halfWidth,
                                 2 * halfHeight);
    }
    else if (mAxis == 0) {
        // Col
        centroidSlice = cv::Rect(centroid.x - halfWidth,
                                 0,
                                 2 * halfWidth,
                                 frame.rows);
    }
    else if (mAxis == 1) {
        // Row
        centroidSlice = cv::Rect(0,
                                 centroid.y - halfHeight,
                                 frame.cols,
                                 2 * halfHeight);
    }

    frame = frame(centroidSlice);

    if (labels.rows > 1 || labels.cols > 1)
        labels = labels(centroidSlice);
}
