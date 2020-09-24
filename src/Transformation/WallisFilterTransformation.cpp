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

#include "Transformation/WallisFilterTransformation.hpp"

const char* N2D2::WallisFilterTransformation::Type = "WallisFilter";

N2D2::WallisFilterTransformation::WallisFilterTransformation(unsigned int size,
                                                             double mean,
                                                             double stdDev)
    : mSize(size),
      mMean(mean),
      mStdDev(stdDev),
      mPerChannel(this, "PerChannel", false)
{
    // ctor
}

N2D2::WallisFilterTransformation::WallisFilterTransformation(
    const WallisFilterTransformation& trans)
    : mSize(trans.mSize),
      mMean(trans.mMean),
      mStdDev(trans.mStdDev),
      mPerChannel(this, "PerChannel", trans.mPerChannel)
{
    // copy-ctor
}

void
N2D2::WallisFilterTransformation::apply(cv::Mat& frame,
                                        cv::Mat& /*labels*/,
                                        std::vector
                                        <std::shared_ptr<ROI> >& /*labelsROI*/,
                                        int /*id*/)
{
    cv::Mat matF;
    frame.convertTo(matF, CV_32F);

    if (mSize > 0) {
        // Get the local mean image
        cv::Mat matMean;
        const cv::Mat meanKernel(
            mSize, mSize, CV_32FC1, cv::Scalar(1.0 / (mSize * mSize)));
        cv::filter2D(matF,
                     matMean,
                     -1,
                     meanKernel,
                     cv::Point(-1, -1),
                     0.0,
#if CV_MAJOR_VERSION >= 3
                     cv::BORDER_REPLICATE);
#else
                     IPL_BORDER_REPLICATE);
#endif

        // Get the standard deviation of the image
        // E[x^2]
        cv::Mat matSquare;
        cv::pow(matF, 2.0, matSquare);
        cv::Mat c1;
        cv::filter2D(matSquare,
                     c1,
                     -1,
                     meanKernel,
                     cv::Point(-1, -1),
                     0.0,
#if CV_MAJOR_VERSION >= 3
                     cv::BORDER_REPLICATE);
#else
                     IPL_BORDER_REPLICATE);
#endif

        // E[x]^2
        cv::Mat c2 = matMean.mul(matMean);

        // max(E[x^2] - E[x]^2, 0)
        cv::Mat matMax;
        cv::max(c1 - c2, 1.0e-6, matMax);  // avoid divide by 0

        // stddev = sqrt(max(E[x^2] - E[x]^2, 0))
        cv::Mat matStdDev;
        cv::sqrt(matMax, matStdDev);

        // Output image
        matF = (matF - matMean) / matStdDev;
    } else {
        const int channels = matF.channels();

        if (mPerChannel) {
            std::vector<cv::Mat> channels;
            cv::split(matF, channels);

            for (int ch = 0; ch < matF.channels(); ++ch)
                channels[ch] = wallisFilter(channels[ch]);

            cv::merge(channels, matF);
        } else {
            cv::Mat frame1 = matF.reshape(1);
            matF = wallisFilter(frame1).reshape(channels);
        }
    }

    frame = matF;
}

cv::Mat N2D2::WallisFilterTransformation::wallisFilter(cv::Mat& mat) const
{
    cv::Scalar mean, stdDev;
    cv::meanStdDev(mat, mean, stdDev);

    mat += (mMean - mean.val[0]);
    mat *= (mStdDev / stdDev.val[0]);

    return mat;
}

N2D2::WallisFilterTransformation::~WallisFilterTransformation() {
    
}
