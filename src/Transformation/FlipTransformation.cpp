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

#include "Transformation/FlipTransformation.hpp"

N2D2::FlipTransformation::FlipTransformation(bool horizontalFlip,
                                             bool verticalFlip)
    : mHorizontalFlip(horizontalFlip),
      mVerticalFlip(verticalFlip),
      mRandomHorizontalFlip(this, "RandomHorizontalFlip", false),
      mRandomVerticalFlip(this, "RandomVerticalFlip", false)
{
    // ctor
}

void N2D2::FlipTransformation::apply(cv::Mat& frame,
                                     cv::Mat& labels,
                                     std::vector
                                     <std::shared_ptr<ROI> >& labelsROI,
                                     int /*id*/)
{
    const bool frameHorizontalFlip
        = (mRandomHorizontalFlip) ? Random::randUniform(0, 1) : mHorizontalFlip;
    const bool frameVerticalFlip
        = (mRandomVerticalFlip) ? Random::randUniform(0, 1) : mVerticalFlip;
    const int flipCode
        = (frameHorizontalFlip && frameVerticalFlip)
              ? -1
              : (frameHorizontalFlip) ? 1 : (frameVerticalFlip) ? 0 : 2;

    flip(frame, flipCode);

    if (labels.rows > 1 || labels.cols > 1)
        flip(labels, flipCode);

    std::for_each(labelsROI.begin(),
                  labelsROI.end(),
                  std::bind(&ROI::flip,
                            std::placeholders::_1,
                            frame.cols,
                            frame.rows,
                            frameHorizontalFlip,
                            frameVerticalFlip));
}

void N2D2::FlipTransformation::reverse(cv::Mat& frame,
                                       cv::Mat& labels,
                                       std::vector
                                       <std::shared_ptr<ROI> >& labelsROI,
                                       int /*id*/)
{
    if (mRandomHorizontalFlip || mRandomVerticalFlip)
        throw std::runtime_error("FlipTransformation::reverse(): cannot "
                                 "reverse random transformation.");

    const int flipCode = (mHorizontalFlip && mVerticalFlip)
                             ? -1
                             : (mHorizontalFlip) ? 1 : (mVerticalFlip) ? 0 : 2;

    if (labels.rows > 1 || labels.cols > 1)
        flip(labels, flipCode);

    std::for_each(labelsROI.begin(),
                  labelsROI.end(),
                  std::bind(&ROI::flip,
                            std::placeholders::_1,
                            frame.cols,
                            frame.rows,
                            mHorizontalFlip,
                            mVerticalFlip));
}

void N2D2::FlipTransformation::flip(cv::Mat& mat, int flipCode) const
{
    if (flipCode != 2) {
        cv::Mat matFlip;
        cv::flip(mat, matFlip, flipCode);
        mat = matFlip;
    }
}
