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

#include "Transformation/RangeClippingTransformation.hpp"

const char* N2D2::RangeClippingTransformation::Type = "RangeClipping";

N2D2::RangeClippingTransformation::RangeClippingTransformation()
    : mRangeMin(this, "RangeMin", 0.0),
      mRangeMax(this, "RangeMax", 0.0),
      mClippingWarn(0)
{
    // ctor
}

N2D2::RangeClippingTransformation::RangeClippingTransformation(
    const RangeClippingTransformation& trans)
    : mRangeMin(this, "RangeMin", trans.mRangeMin),
      mRangeMax(this, "RangeMax", trans.mRangeMax),
      mClippingWarn(trans.mClippingWarn)
{
    // copy-ctor
}

void
N2D2::RangeClippingTransformation::apply(cv::Mat& frame,
                                         cv::Mat& /*labels*/,
                                         std::vector
                                         <std::shared_ptr<ROI> >& /*labelsROI*/,
                                         int /*id*/)
{
    const int channels = frame.channels();
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

    // Search min and max value in the image
    double minVal, maxVal;
    cv::minMaxLoc(frame.reshape(1), &minVal, &maxVal);

    const double minRangeVal = (mRangeMin > 0) ? mRangeMin : minVal;
    const double maxRangeVal = (mRangeMax > 0) ? mRangeMax : maxVal;

    if (minVal < minRangeVal || maxVal > maxRangeVal) {
        const int warnLimit = 5;

        if (mClippingWarn < warnLimit) {
            std::cout << Utils::cwarning
                    << "Warning: clipping image value range, from [" << minVal
                    << ", " << maxVal << "]"
                                        " to [" << minRangeVal << ", "
                    << maxRangeVal << "]." << Utils::cdef << std::endl;

            ++mClippingWarn;

            if (mClippingWarn == warnLimit) {
                std::cout << Utils::cwarning
                    << "Future clipping warning will be ignored!" << Utils::cdef
                    << std::endl;
            }
        }
    }

    // Range clipping such that:
    // minRangeVal becomes 0
    // maxRangeVal becomes maxValue
    //
    // new_value = scale*old_value + shift with
    // scale = maxValue/(maxRangeVal - minRangeVal)
    // shift = -maxValue*minRangeVal/(maxRangeVal - minRangeVal) =
    // -minRangeVal*a
    //
    // minRangeVal: maxValue/(maxRangeVal - minRangeVal)*minRangeVal
    // -maxValue*minRangeVal/(maxRangeVal - minRangeVal) = 0
    // maxRangeVal: maxValue/(maxRangeVal - minRangeVal)*maxRangeVal
    // -maxValue*minRangeVal/(maxRangeVal - minRangeVal) = maxValue

    const double scale = maxValue * (((maxRangeVal - minRangeVal) > 0)
                                         ? 1.0 / (maxRangeVal - minRangeVal)
                                         : 0.0);
    const double shift = -minRangeVal * scale;

    cv::Mat frameNormalized;
    frame.reshape(1).convertTo(frameNormalized, -1, scale, shift);

    if (maxValue == 1.0) {
        // Clip float in [0,1] (no saturate_cast<> in convertTo())
        frameNormalized = cv::min(cv::max(frameNormalized, 0.0), maxValue);
    }

    frame = frameNormalized.reshape(channels);
}

N2D2::RangeClippingTransformation::~RangeClippingTransformation() {
    
}
