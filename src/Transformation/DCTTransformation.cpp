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

#include "Transformation/DCTTransformation.hpp"

const char* N2D2::DCTTransformation::Type = "DCT";

N2D2::DCTTransformation::DCTTransformation(bool twoDimensional)
    : mTwoDimensional(twoDimensional)
{
    // ctor
}

void N2D2::DCTTransformation::apply(cv::Mat& frame,
                                    cv::Mat& /*labels*/,
                                    std::vector
                                    <std::shared_ptr<ROI> >& /*labelsROI*/,
                                    int /*id*/)
{
    if (frame.channels() != 1)
        throw std::runtime_error(
            "DCTTransformation: require single channel input");

    // expand input image to optimal size
    cv::Mat padded;
    const int m = getOptimalDCTSize(frame.rows);
    const int n = getOptimalDCTSize(frame.cols);

    // on the border add zero values
    cv::copyMakeBorder(frame,
                       padded,
                       0,
                       m - frame.rows,
                       0,
                       n - frame.cols,
                       cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));

    cv::dct(padded,
            frame,
            (!mTwoDimensional) ? cv::DCT_ROWS : 0);
}

size_t N2D2::DCTTransformation::getOptimalDCTSize(size_t N) const {
    return 2 * cv::getOptimalDFTSize((N + 1) / 2);
}
