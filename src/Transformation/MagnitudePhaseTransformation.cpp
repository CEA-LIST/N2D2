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

#include "Transformation/MagnitudePhaseTransformation.hpp"

N2D2::MagnitudePhaseTransformation::MagnitudePhaseTransformation(bool logScale)
    : mLogScale(logScale)
{
    // ctor
}

void N2D2::MagnitudePhaseTransformation::apply(
    cv::Mat& frame,
    cv::Mat& /*labels*/,
    std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
    int /*id*/)
{
    if (frame.channels() != 2)
        throw std::runtime_error("MagnitudePhaseTransformation: require two "
                                 "channels input (real and imag parts)");

    std::vector<cv::Mat> planes;
    cv::split(frame, planes);

    // Magnitude
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    if (mLogScale) {
        mag += cv::Scalar::all(1);
        cv::log(mag, mag);
    }

    // Phase
    // cv::phase() has poor precision and is not between ]-pi:pi] in OpenCV
    // 2.0.0
    // cv::phase(planes[0], planes[1], planes[1]);

    for (int i = 0; i < planes[0].rows; ++i) {
        double* rowPtrReal = planes[0].ptr<double>(i);
        double* rowPtrImag = planes[1].ptr<double>(i);

        for (int j = 0; j < planes[0].cols; ++j)
            rowPtrImag[j] = std::atan2(rowPtrImag[j], rowPtrReal[j]);
    }

    // Merge
    planes[0] = mag;
    cv::merge(planes, frame);
}
