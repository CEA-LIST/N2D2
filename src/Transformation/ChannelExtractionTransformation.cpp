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

#include "Transformation/ChannelExtractionTransformation.hpp"

const char* N2D2::ChannelExtractionTransformation::Type = "ChannelExtraction";

N2D2::ChannelExtractionTransformation::ChannelExtractionTransformation(
    Channel channel)
    : mChannel(channel)
{
    // ctor
}

void N2D2::ChannelExtractionTransformation::apply(
    cv::Mat& frame,
    cv::Mat& /*labels*/,
    std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
    int /*id*/)
{
    if (!(frame.channels() > 1))
        return;

    cv::Mat frameCvt;

    if (mChannel == Hue || mChannel == Saturation || mChannel == Value) {
        // RGB to HSV conversion
#if CV_MAJOR_VERSION >= 3
        cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2HSV);
#else
        cv::cvtColor(frame, frameCvt, CV_BGR2HSV);
#endif
    } else if (mChannel == Y || mChannel == Cb || mChannel == Cr) {
        // RGB to YUV conversion
#if CV_MAJOR_VERSION >= 3
        cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2YCrCb);
#else
        cv::cvtColor(frame, frameCvt, CV_BGR2YCrCb);
#endif
    } else
        frameCvt = frame;

    std::vector<cv::Mat> channels;
    cv::split(frameCvt, channels);

    switch (mChannel) {
    case Blue:
    case Y:
        frameCvt = channels[0];
        break;

    case Hue:
        if (frame.depth() == CV_8U) {
            // 8 bits Hue is between 0 and 180 with OpenCV
            channels[0].convertTo(frameCvt, CV_8UC1, 255.0 / 180.0);
        } else
            frameCvt = channels[0];
        break;

    case Green:
    case Saturation:
    case Cr:
        frameCvt = channels[1];
        break;

    case Red:
    case Value:
    case Cb:
        frameCvt = channels[2];
        break;

    case Gray:
    default:
#if CV_MAJOR_VERSION >= 3
        cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2GRAY);
#else
        cv::cvtColor(frame, frameCvt, CV_BGR2GRAY);
#endif
        break;
    }

    frame = frameCvt;
}
