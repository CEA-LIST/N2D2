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

#include "Transformation/FilterTransformation.hpp"

namespace N2D2 {
const FilterTransformation
FilterTransformationLaplacian(Kernel<double>("0 -1 0 -1 4 -1 0 -1 0"), 0.0);
const FilterTransformation FilterTransformationAerPositive(Kernel<double>("1"),
                                                           0.0);
const FilterTransformation FilterTransformationAerNegative(Kernel<double>("-1"),
                                                           0.5);
}

N2D2::FilterTransformation::FilterTransformation(const Kernel<double>& kernel,
                                                 double orientation)
    : mKernel(kernel), mOrientation(orientation)
{
    // ctor
    if (mKernel.empty())
        throw std::runtime_error(
            "FilterTransformation: filter's kernel is empty!");
}

void N2D2::FilterTransformation::apply(cv::Mat& frame,
                                       cv::Mat& /*labels*/,
                                       std::vector
                                       <std::shared_ptr<ROI> >& /*labelsROI*/,
                                       int /*id*/)
{
    // Kernel needs to be flipped for cv::filter2D() to perform a real
    // convolution
    // http://docs.opencv.org/modules/imgproc/doc/filtering.html#filter2d
    // cv::Mat kernel = cv::Mat(filter_.kernel()).clone();   // clone() to make
    // sure to copy the data,
    // else the kernel itself will be changed
    // cv::flip(kernel, kernel, -1);
    // No flip to be equivalent to Magick::Image::convolve()
    cv::filter2D(frame, frame, -1, (cv::Mat)mKernel, cv::Point(-1, -1), 0.0);
}

namespace N2D2 {
FilterTransformation operator-(const FilterTransformation& filter)
{
    return FilterTransformation(-filter.mKernel,
                                std::fmod(filter.mOrientation + 0.5, 1.0));
}
}
