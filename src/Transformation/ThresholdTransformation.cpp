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

#include "Transformation/ThresholdTransformation.hpp"

N2D2::ThresholdTransformation::ThresholdTransformation(double threshold,
                                                       bool otsuMethod)
    : mThreshold(threshold),
      mOtsuMethod(otsuMethod),
      mOperation(this, "Operation", Binary),
      mMaxValue(this, "MaxValue", 1.0)
{
    // ctor
}

void
N2D2::ThresholdTransformation::apply(cv::Mat& frame,
                                     cv::Mat& /*labels*/,
                                     std::vector
                                     <std::shared_ptr<ROI> >& /*labelsROI*/,
                                     int /*id*/)
{
    int type = (mOperation == BinaryInverted)
                   ? cv::THRESH_BINARY_INV
                   : (mOperation == Truncate)
                         ? cv::THRESH_TRUNC
                         : (mOperation == ToZero)
                               ? cv::THRESH_TOZERO
                               : (mOperation == ToZeroInverted)
                                     ? cv::THRESH_TOZERO_INV
                                     : cv::THRESH_BINARY;

    if (mOtsuMethod)
        type |= cv::THRESH_OTSU;

    if (frame.channels() > 1) {
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);

        for (int ch = 0; ch < frame.channels(); ++ch)
            cv::threshold(
                channels[ch], channels[ch], mThreshold, mMaxValue, type);

        cv::merge(channels, frame);
    } else
        cv::threshold(frame, frame, mThreshold, mMaxValue, type);
}
