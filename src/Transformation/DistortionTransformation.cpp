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

#include "Transformation/DistortionTransformation.hpp"

const char* N2D2::DistortionTransformation::Type = "Distortion";

N2D2::DistortionTransformation::DistortionTransformation()
    : mElasticGaussianSize(this, "ElasticGaussianSize", 15U),
      mElasticSigma(this, "ElasticSigma", 6.0),
      mElasticScaling(this, "ElasticScaling", 0.0),
      mScaling(this, "Scaling", 0.0),
      mRotation(this, "Rotation", 0.0),
      mIgnoreMissingData(this, "IgnoreMissingData", true)
{
    // ctor
}

N2D2::DistortionTransformation::DistortionTransformation(
    const DistortionTransformation& trans)
    : mElasticGaussianSize(this, "ElasticGaussianSize",
                           trans.mElasticGaussianSize),
      mElasticSigma(this, "ElasticSigma", trans.mElasticSigma),
      mElasticScaling(this, "ElasticScaling", trans.mElasticScaling),
      mScaling(this, "Scaling", trans.mScaling),
      mRotation(this, "Rotation", trans.mRotation),
      mIgnoreMissingData(this, "IgnoreMissingData", trans.mIgnoreMissingData)
{
    // copy-ctor
}

void
N2D2::DistortionTransformation::apply(cv::Mat& frame,
                                      cv::Mat& labels,
                                      std::vector
                                      <std::shared_ptr<ROI> >& /*labelsROI*/,
                                      int /*id*/)
{
    const unsigned int sizeX = frame.cols;
    const unsigned int sizeY = frame.rows;
    // const unsigned int size = sizeX*sizeY;
    const unsigned int gaussianSizeX = mElasticGaussianSize;
    const unsigned int gaussianSizeY = mElasticGaussianSize;

    // Elastic scaling init
    Matrix<double> uniformX;
    Matrix<double> uniformY;

    if (mElasticScaling > 0.0) {
        uniformX.resize(sizeY, sizeX);
        uniformY.resize(sizeY, sizeX);

        for (unsigned int index = 0, size = uniformX.size(); index < size;
             ++index) {
            uniformX(index) = Random::randUniform(-1.0, 1.0);
            uniformY(index) = Random::randUniform(-1.0, 1.0);
        }

        if (mKernel.empty()) {
#pragma omp critical(DistortionTransformation__apply)
            if (mKernel.empty())
                mKernel = GaussianKernel
                    <double>(gaussianSizeX, gaussianSizeY, mElasticSigma);
        }
    }

    const int centerX = sizeX / 2;
    const int centerY = sizeY / 2;
    const int kCenterX = gaussianSizeX / 2;
    const int kCenterY = gaussianSizeY / 2;

    // Scaling init
    const double scaleX = (mScaling / 100.0) * Random::randUniform(-1.0, 1.0);
    const double scaleY = (mScaling / 100.0) * Random::randUniform(-1.0, 1.0);

    // Rotation init
    const double rotate
        = Utils::degToRad(mRotation * Random::randUniform(-1.0, 1.0));
    const double rotateCos = std::cos(rotate);
    const double rotateSin = std::sin(rotate);

    Matrix<double> dispX(sizeY, sizeX);
    Matrix<double> dispY(sizeY, sizeX);
    /*
    #if defined(_OPENMP) && _OPENMP >= 200805
        #pragma omp parallel for collapse(2) if (size > 256)
    #else
        #pragma omp parallel for if (sizeY > 16 && size > 256)
    #endif
    */
    for (int y = 0; y < (int)sizeY; ++y) { // rows
        for (unsigned int x = 0; x < sizeX; ++x) { // columns
            // Elastic scaling
            double vX = 0.0;
            double vY = 0.0;

            if (mElasticScaling > 0.0) {
                for (unsigned int kx = 0; kx < gaussianSizeX;
                     ++kx) { // kernel columns
                    const unsigned int fkx
                        = gaussianSizeX - 1
                          - (int)kx; // column index of flipped kernel

                    assert(fkx < gaussianSizeX);

                    for (unsigned int ky = 0; ky < gaussianSizeY;
                         ++ky) { // kernel rows
                        const unsigned int fky
                            = gaussianSizeY - 1
                              - (int)ky; // row index of flipped kernel

                        assert(fky < gaussianSizeY);

                        // index of input signal, used for checking boundary
                        const int mx = x + (kx - kCenterX);
                        const int my = y + (ky - kCenterY);

                        // ignore input samples which are out of bound
                        if (mx >= 0 && mx < (int)sizeX && my >= 0
                            && my < (int)sizeY) {
                            vX += uniformX(my, mx) * mKernel(fky, fkx);
                            vY += uniformY(my, mx) * mKernel(fky, fkx);
                        }
                    }
                }

                vX *= mElasticScaling;
                vY *= mElasticScaling;
            }

            // Scaling
            if (mScaling > 0.0) {
                vX += scaleX * ((int)x - centerX);
                vY += scaleY * ((int)y - centerY);
            }

            // Rotation
            if (mRotation > 0.0) {
                vX += ((int)x - centerX) * (rotateCos - 1.0)
                      + ((int)y - centerY) * rotateSin;
                vY += ((int)y - centerY) * (rotateCos - 1.0)
                      - ((int)x - centerX) * rotateSin;
            }

            dispX(y, x) = vX;
            dispY(y, x) = vY;
        }
    }

    const DistortionMap_T distortionMap = std::make_pair(dispX, dispY);

    applyDistortion(frame, distortionMap, false);

    if (labels.rows > 1 || labels.cols > 1)
        applyDistortion(labels, distortionMap, true);
}

void N2D2::DistortionTransformation::applyDistortion(cv::Mat& mat,
                                                     const DistortionMap_T
                                                     & distortionMap,
                                                     bool nearestNeighbor) const
{
    switch (mat.depth()) {
    case CV_8U:
        applyDistortionType<unsigned char>(mat, distortionMap, nearestNeighbor);
        break;
    case CV_8S:
        applyDistortionType<char>(mat, distortionMap, nearestNeighbor);
        break;
    case CV_16U:
        applyDistortionType
            <unsigned short>(mat, distortionMap, nearestNeighbor);
        break;
    case CV_16S:
        applyDistortionType<short>(mat, distortionMap, nearestNeighbor);
        break;
    case CV_32S:
        applyDistortionType<int>(mat, distortionMap, nearestNeighbor);
        break;
    case CV_32F:
        applyDistortionType<float>(mat, distortionMap, nearestNeighbor);
        break;
    case CV_64F:
        applyDistortionType<double>(mat, distortionMap, nearestNeighbor);
        break;
    default:
        throw std::runtime_error("Cannot apply distortion: incompatible type.");
    }
}

N2D2::DistortionTransformation::~DistortionTransformation() {
    
}
