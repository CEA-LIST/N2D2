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

#include "Transformation/CompressionNoiseTransformation.hpp"

const char* N2D2::CompressionNoiseTransformation::Type = "CompressionNoise";

N2D2::CompressionNoiseTransformation::CompressionNoiseTransformation()
    : mCompressionRange(this, "CompressionRange",
                           std::vector<int>({0, 100}))
{
    // ctor
}

N2D2::CompressionNoiseTransformation::CompressionNoiseTransformation(
    const CompressionNoiseTransformation& trans)
    : mCompressionRange(this, "CompressionRange", trans.mCompressionRange)
{
    // copy-ctor
}

void
N2D2::CompressionNoiseTransformation::apply(cv::Mat& frame,
                                     cv::Mat& /*labels*/,
                                     std::vector
                                     <std::shared_ptr<ROI> >& /*labelsROI*/,
                                     int /*id*/)
{
    if (mCompressionRange->size() != 2) {
        throw std::runtime_error("CompressionNoiseTransformation::apply(): "
                                 "mCompressionRange must have two values "
                                 "(\"min max\")");
    }

    const int quality
        = 100 - Random::randUniform(*(mCompressionRange->begin()),
                                    *(mCompressionRange->begin() + 1));

    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(quality);

    std::vector<unsigned char> buffer;
    cv::imencode(".jpg", frame, buffer, params);

#if CV_MAJOR_VERSION >= 3
    frame = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
#else
    frame = cv::imdecode(buffer, CV_LOAD_IMAGE_UNCHANGED);
#endif
}
