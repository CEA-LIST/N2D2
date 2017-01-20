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

#include "Transformation/NormalizeTransformation.hpp"

N2D2::NormalizeTransformation::NormalizeTransformation()
    : mNorm(this, "Norm", MinMax),
      mNormValue(this, "NormValue", 1.0),
      mNormMin(this, "NormMin", 0.0),
      mNormMax(this, "NormMax", 1.0),
      mPerChannel(this, "PerChannel", false)
{
    // ctor
}

void
N2D2::NormalizeTransformation::apply(cv::Mat& frame,
                                     cv::Mat& /*labels*/,
                                     std::vector
                                     <std::shared_ptr<ROI> >& /*labelsROI*/,
                                     int /*id*/)
{
    const int channels = frame.channels();

    if (mPerChannel) {
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);

        for (int ch = 0; ch < frame.channels(); ++ch)
            channels[ch] = normalize(channels[ch]);

        cv::merge(channels, frame);
    } else {
        cv::Mat frame1 = frame.reshape(1);
        frame = normalize(frame1).reshape(channels);
    }
}

cv::Mat N2D2::NormalizeTransformation::normalize(cv::Mat& mat) const
{
    double maxValue = 1.0;

    switch (mat.depth()) {
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

    cv::Mat matNorm;

    if (mNorm == MinMax)
        cv::normalize(mat,
                      matNorm,
                      mNormMin * maxValue,
                      mNormMax * maxValue,
                      cv::NORM_MINMAX);
    else {
        const int normType = (mNorm == L1) ? cv::NORM_L1 : (mNorm == L2)
                                                               ? cv::NORM_L2
                                                               : cv::NORM_INF;

        cv::Mat mat64F;
        mat.convertTo(mat64F, CV_64F, 1.0 / maxValue);
        cv::normalize(mat64F, matNorm, mNormValue, 0.0, normType);
    }

    return matNorm;
}
