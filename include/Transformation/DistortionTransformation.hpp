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

#ifndef N2D2_DISTORTIONTRANSFORMATION_H
#define N2D2_DISTORTIONTRANSFORMATION_H

#include "Transformation/Transformation.hpp"
#include "utils/Kernel.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class DistortionTransformation : public Transformation {
public:
    using Transformation::apply;

    typedef std::pair<Matrix<double>, Matrix<double> > DistortionMap_T;

    static const char* Type;

    DistortionTransformation();
    DistortionTransformation(const DistortionTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<DistortionTransformation> clone() const
    {
        return std::shared_ptr<DistortionTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~DistortionTransformation();
    
    unsigned int getElasticGaussianSize(){
        return mElasticGaussianSize;
    };
    double getElasticSigma(){
        return mElasticSigma;
    };
    double getElasticScaling(){
        return mElasticScaling;
    };
    double getScaling(){
        return mScaling;
    };
    double getRotation(){
        return mRotation;
    };
    bool getIgnoreMissingData(){
        return mIgnoreMissingData;
    };

private:
    virtual DistortionTransformation* doClone() const
    {
        return new DistortionTransformation(*this);
    }
    void applyDistortion(cv::Mat& mat,
                         const DistortionMap_T& distortionMap,
                         bool nearestNeighbor) const;
    template <class T>
    void applyDistortionType(cv::Mat& mat,
                             const DistortionMap_T& distortionMap,
                             bool nearestNeighbor) const;

    Matrix<double> mKernel;

    Parameter<unsigned int> mElasticGaussianSize;
    Parameter<double> mElasticSigma;
    Parameter<double> mElasticScaling;
    Parameter<double> mScaling;
    Parameter<double> mRotation;
    Parameter<bool> mIgnoreMissingData;
};
}

template <class T>
void
N2D2::DistortionTransformation::applyDistortionType(cv::Mat& mat,
                                                    const DistortionMap_T
                                                    & distortionMap,
                                                    bool nearestNeighbor) const
{
    const int rows = mat.rows;
    const int cols = mat.cols;
    // const int size = rows*cols;

    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    std::vector<cv::Mat> distortedChannels;

    for (int ch = 0; ch < mat.channels(); ++ch)
        distortedChannels.push_back(
            cv::Mat(mat.rows, mat.cols, channels[ch].type(), cv::Scalar(0)));
    /*
    #if defined(_OPENMP) && _OPENMP >= 200805
        #pragma omp parallel for collapse(2) if (size > 256)
    #else
        #pragma omp parallel for if (rows > 16 && size > 256)
    #endif
    */
    for (int i = 0; i < rows; ++i) { // rows
        for (int j = 0; j < cols; ++j) { // columns
            const double isrc_abs = (double)i - distortionMap.second(i, j);
            const double jsrc_abs = (double)j - distortionMap.first(i, j);

            if (mIgnoreMissingData && nearestNeighbor &&
                (isrc_abs < 0.0 || isrc_abs > ((double)rows - 2.0)
                || jsrc_abs < 0.0 || jsrc_abs > ((double)cols - 2.0)))
            {
                for (int ch = 0; ch < mat.channels(); ++ch)
                    distortedChannels[ch].at<T>(i, j) = -1;

                continue;
            }

            const double isrc = Utils::clamp(isrc_abs, 0.0, (double)rows - 2.0);
            const double jsrc = Utils::clamp(jsrc_abs, 0.0, (double)cols - 2.0);
            const int isrc0 = (int)isrc;
            const int jsrc0 = (int)jsrc;

            // Debug
            // std::cout << "(" << i << "," << j << ") " <<
            // distortionMap.second(i,j)
            // << " -> (" << isrc << "," << jsrc << ")" << std::endl;

            const double di = isrc - isrc0;
            const double dj = jsrc - jsrc0;

            const double d00 = (1.0 - di) * (1.0 - dj);
            const double d10 = di * (1.0 - dj);
            const double d01 = (1.0 - di) * dj;
            const double d11 = di * dj;

            if (nearestNeighbor) {
                // Perform nearest neighbor interpolation
                const double d00_d10 = std::max(d00, d10);
                const double d01_d11 = std::max(d01, d11);

                for (int ch = 0; ch < mat.channels(); ++ch) {
                    distortedChannels[ch].at<T>(i, j)
                        = (d00_d10 > d01_d11)
                              ? ((d00 > d10)
                                     ? channels[ch].at<T>(isrc0, jsrc0)
                                     : channels[ch].at<T>(isrc0 + 1, jsrc0))
                              : ((d01 > d11)
                                     ? channels[ch].at<T>(isrc0, jsrc0 + 1)
                                     : channels[ch].at
                                       <T>(isrc0 + 1, jsrc0 + 1));
                }
            } else {
                // Perform bilinear interpolation
                for (int ch = 0; ch < mat.channels(); ++ch) {
                    distortedChannels[ch].at<T>(i, j)
                        = channels[ch].at<T>(isrc0, jsrc0) * d00
                          + channels[ch].at<T>(isrc0 + 1, jsrc0) * d10
                          + channels[ch].at<T>(isrc0, jsrc0 + 1) * d01
                          + channels[ch].at<T>(isrc0 + 1, jsrc0 + 1) * d11;
                }
            }
        }
    }

    cv::merge(distortedChannels, mat);
}

#endif // N2D2_DISTORTIONTRANSFORMATION_H
