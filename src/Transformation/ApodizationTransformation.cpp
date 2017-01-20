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

#include "Transformation/ApodizationTransformation.hpp"

N2D2::ApodizationTransformation::ApodizationTransformation(const WindowFunction
                                                           <double>& func,
                                                           unsigned int size)
    : mWindow(func(size))
{
    // ctor
}

void
N2D2::ApodizationTransformation::apply(cv::Mat& frame,
                                       cv::Mat& /*labels*/,
                                       std::vector
                                       <std::shared_ptr<ROI> >& /*labelsROI*/,
                                       int /*id*/)
{
    if (frame.cols != (int)mWindow.size())
        throw std::runtime_error("ApodizationTransformation: input size does "
                                 "not match apodization window size");

    switch (frame.depth()) {
    case CV_8U:
        applyApodization<unsigned char>(frame);
        break;
    case CV_8S:
        applyApodization<char>(frame);
        break;
    case CV_16U:
        applyApodization<unsigned short>(frame);
        break;
    case CV_16S:
        applyApodization<short>(frame);
        break;
    case CV_32S:
        applyApodization<int>(frame);
        break;
    case CV_32F:
        applyApodization<float>(frame);
        break;
    case CV_64F:
        applyApodization<double>(frame);
        break;
    default:
        throw std::runtime_error(
            "Cannot apply apodization: incompatible type.");
    }
}
