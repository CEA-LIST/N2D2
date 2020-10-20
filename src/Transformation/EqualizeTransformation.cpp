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

#include "Transformation/EqualizeTransformation.hpp"

const char* N2D2::EqualizeTransformation::Type = "Equalize";

N2D2::EqualizeTransformation::EqualizeTransformation()
    : mMethod(this, "Method", Standard),
      mCLAHE_ClipLimit(this, "CLAHE_ClipLimit", 40.0),
      mCLAHE_GridSize(this, "CLAHE_GridSize", 8U)
{
    // ctor
}

N2D2::EqualizeTransformation::EqualizeTransformation(
    const EqualizeTransformation& trans)
    : mMethod(this, "Method", trans.mMethod),
      mCLAHE_ClipLimit(this, "CLAHE_ClipLimit", trans.mCLAHE_ClipLimit),
      mCLAHE_GridSize(this, "CLAHE_GridSize", trans.mCLAHE_GridSize)
{
    // copy-ctor
}

void N2D2::EqualizeTransformation::apply(cv::Mat& frame,
                                         cv::Mat& /*labels*/,
                                         std::vector
                                         <std::shared_ptr<ROI> >& /*labelsROI*/,
                                         int /*id*/)
{
    cv::Mat frameLab;
    std::vector<cv::Mat> channels;

    if (frame.channels() > 1) {
#if CV_MAJOR_VERSION >= 3
        cv::cvtColor(frame, frameLab, cv::COLOR_BGR2Lab);
#else
        cv::cvtColor(frame, frameLab, CV_BGR2Lab);
#endif
        cv::split(frameLab, channels);
    } else
        channels.push_back(frame);

    if (mMethod == Standard)
        cv::equalizeHist(channels[0], channels[0]);
    else {
#if CV_MAJOR_VERSION > 2 || (CV_MAJOR_VERSION == 2 && \
    (CV_MINOR_VERSION > 4 \
        || (CV_MINOR_VERSION == 4 && CV_SUBMINOR_VERSION >= 5)))
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
            mCLAHE_ClipLimit, cv::Size(mCLAHE_GridSize, mCLAHE_GridSize));
        clahe->apply(channels[0], channels[0]);
#else
        throw std::runtime_error("EqualizeTransformation::apply(): Adaptive "
                                 "method requires at least OpenCV 2.4.5,"
                                 " but compiled version is "
                                 + std::string(CV_VERSION));
#endif
    }

    if (frame.channels() > 1) {
        cv::merge(channels, frameLab);
#if CV_MAJOR_VERSION >= 3
        cv::cvtColor(frameLab, frame, cv::COLOR_Lab2BGR);
#else
        cv::cvtColor(frameLab, frame, CV_Lab2BGR);
#endif
    } else
        frame = channels[0];
}

N2D2::EqualizeTransformation::~EqualizeTransformation() {
    
}
