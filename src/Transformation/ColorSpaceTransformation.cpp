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
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_GRAY2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_GRAY2BGR);
#endif
            frame = frameCvt;
        } else if (frame.channels() == 4) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_BGRA2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_BGRA2BGR);
#endif
            frame = frameCvt;
        }
    }
    else if (mColorSpace == RGB) {
        if (frame.channels() == 1) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_GRAY2RGB);
#else
            cv::cvtColor(frame, frameCvt, CV_GRAY2RGB);
#endif
            frame = frameCvt;
        }
        else if (frame.channels() == 3) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2RGB);
#else
            cv::cvtColor(frame, frameCvt, CV_BGR2RGB);
#endif
            frame = frameCvt;
        }
        else if (frame.channels() == 4) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_RGBA2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_RGBA2BGR);
#endif
            frame = frameCvt;
        }

    }

    else if (frame.channels() > 1) {
        if (mColorSpace == HSV) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2HSV);
#else
            cv::cvtColor(frame, frameCvt, CV_BGR2HSV);
#endif
        }
        else if (mColorSpace == HLS) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2HLS);
#else
            cv::cvtColor(frame, frameCvt, CV_BGR2HLS);
#endif
        }
        else if (mColorSpace == YCrCb) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2YCrCb);
#else
            cv::cvtColor(frame, frameCvt, CV_BGR2YCrCb);
#endif
        }
        else if (mColorSpace == CIELab) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2Lab);
#else
            cv::cvtColor(frame, frameCvt, CV_BGR2Lab);
#endif
        }
        else if (mColorSpace == CIELuv) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_BGR2Luv);
#else
            cv::cvtColor(frame, frameCvt, CV_BGR2Luv);
#endif
        }
        // RGB to
        else if (mColorSpace == RGB_to_BGR) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_RGB2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_RGB2BGR);
#endif
        }
        else if (mColorSpace == RGB_to_HSV) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_RGB2HSV);
#else
            cv::cvtColor(frame, frameCvt, CV_RGB2HSV);
#endif
        }
        else if (mColorSpace == RGB_to_HLS) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_RGB2HLS);
#else
            cv::cvtColor(frame, frameCvt, CV_RGB2HLS);
#endif
        }
        else if (mColorSpace == RGB_to_YCrCb) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_RGB2YCrCb);
#else
            cv::cvtColor(frame, frameCvt, CV_RGB2YCrCb);
#endif
        }
        else if (mColorSpace == RGB_to_CIELab) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_RGB2Lab);
#else
            cv::cvtColor(frame, frameCvt, CV_RGB2Lab);
#endif
        }
        else if (mColorSpace == RGB_to_CIELuv) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_RGB2Luv);
#else
            cv::cvtColor(frame, frameCvt, CV_RGB2Luv);
#endif
        }
        // HSV to
        else if (mColorSpace == HSV_to_BGR) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_HSV2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_HSV2BGR);
#endif
        }
        else if (mColorSpace == HSV_to_RGB) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_HSV2RGB);
#else
            cv::cvtColor(frame, frameCvt, CV_HSV2RGB);
#endif
        }
        // HLS to
        else if (mColorSpace == HLS_to_BGR) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_HLS2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_HLS2BGR);
#endif
        }
        else if (mColorSpace == HLS_to_RGB) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_HLS2RGB);
#else
            cv::cvtColor(frame, frameCvt, CV_HLS2RGB);
#endif
        }
        // YCrCb to
        else if (mColorSpace == YCrCb_to_BGR) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_YCrCb2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_YCrCb2BGR);
#endif
        }
        else if (mColorSpace == YCrCb_to_RGB) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_YCrCb2RGB);
#else
            cv::cvtColor(frame, frameCvt, CV_YCrCb2RGB);
#endif
        }
        // CIELab to
        else if (mColorSpace == CIELab_to_BGR) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_Lab2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_Lab2BGR);
#endif
        }
        else if (mColorSpace == CIELab_to_RGB) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_Lab2RGB);
#else
            cv::cvtColor(frame, frameCvt, CV_Lab2RGB);
#endif
        }
        // CIELuv to
        else if (mColorSpace == CIELuv_to_BGR) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_Luv2BGR);
#else
            cv::cvtColor(frame, frameCvt, CV_Luv2BGR);
#endif
        }
        else if (mColorSpace == CIELuv_to_RGB) {
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(frame, frameCvt, cv::COLOR_Luv2RGB);
#else
            cv::cvtColor(frame, frameCvt, CV_Luv2RGB);
#endif
        }

        frame = frameCvt;
    }
}
