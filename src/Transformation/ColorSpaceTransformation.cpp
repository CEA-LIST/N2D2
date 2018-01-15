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

#include "Transformation/ColorSpaceTransformation.hpp"

N2D2::ColorSpaceTransformation::ColorSpaceTransformation(ColorSpace colorSpace)
    : mColorSpace(colorSpace)
{
    // ctor
}

void
N2D2::ColorSpaceTransformation::apply(cv::Mat& frame,
                                      cv::Mat& /*labels*/,
                                      std::vector
                                      <std::shared_ptr<ROI> >& /*labelsROI*/,
                                      int /*id*/)
{
    cv::Mat frameCvt;

    if (mColorSpace == BGR) {
        if (frame.channels() == 1) {
            cv::cvtColor(frame, frameCvt, CV_GRAY2BGR);
            frame = frameCvt;
        } else if (frame.channels() == 4) {
            cv::cvtColor(frame, frameCvt, CV_BGRA2BGR);
            frame = frameCvt;
        }
    }
    else if (mColorSpace == RGB) {
        if (frame.channels() == 1) {
            cv::cvtColor(frame, frameCvt, CV_GRAY2RGB);
            frame = frameCvt;
        }
        else if (frame.channels() == 3) {
            cv::cvtColor(frame, frameCvt, CV_BGR2RGB);
            frame = frameCvt;
        }
        else if (frame.channels() == 4) {
            cv::cvtColor(frame, frameCvt, CV_RGBA2BGR);
            frame = frameCvt;
        }

    }

    else if (frame.channels() > 1) {
        if (mColorSpace == HSV)
            cv::cvtColor(frame, frameCvt, CV_BGR2HSV);
        else if (mColorSpace == HLS)
            cv::cvtColor(frame, frameCvt, CV_BGR2HLS);
        else if (mColorSpace == YCrCb)
            cv::cvtColor(frame, frameCvt, CV_BGR2YCrCb);
        else if (mColorSpace == CIELab)
            cv::cvtColor(frame, frameCvt, CV_BGR2Lab);
        else if (mColorSpace == CIELuv)
            cv::cvtColor(frame, frameCvt, CV_BGR2Luv);

        frame = frameCvt;
    }
}
