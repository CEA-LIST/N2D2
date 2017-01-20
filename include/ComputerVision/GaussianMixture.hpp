/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_COMPUTERVISION_GAUSSIANMIXTURE_H
#define N2D2_COMPUTERVISION_GAUSSIANMIXTURE_H

#include <tuple>
#include <vector>

#include "ComputerVision/ROI.hpp"
#include "containers/Matrix.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/Parameterizable.hpp"

namespace N2D2 {
namespace ComputerVision {
    /**
     * Implementation of the gaussian mixture model.
     * See C. Stauffer and W.E.L. Grimson, "Adaptive background mixture models
     * for real-time tracking," Computer Vision and
     * Pattern Recognition, 1999. IEEE Computer Society Conference on. , vol.2,
     * no., pp.,252 Vol. 2, 1999.
     * And S. Atev, O. Masoud and N. Papanikolopoulos, "Practical mixtures of
     * Gaussians with brightness monitoring," Intelligent
     * Transportation Systems, 2004. Proceedings. The 7th International IEEE
     * Conference on , vol., no., pp.423,428, 3-6 Oct.
     * 2004.
    */
    class GaussianMixture : public Parameterizable {
    public:
        struct GaussianModel_T {
            GaussianModel_T(double w_, double mu_, double sigma_)
                : w(w_), mu(mu_), sigma(sigma_) {};

            double w;
            double mu;
            double sigma;
        };

        GaussianMixture(unsigned int k);
        Matrix<unsigned char> getForeground(const Matrix<double>& frame);
        void excludeRoi(const ROI::Roi_T& roi);
        void updateModel(const Matrix<double>& frame);
        Matrix<double> getBaseBackground(unsigned int level = 0) const;
        void getPixelModel(unsigned int x, unsigned int y) const;
        void load(const std::string& fileName, bool ignoreNotExists = false);
        void save(const std::string& fileName) const;

    private:
        static bool modelCompare(const GaussianModel_T& lhs,
                                 const GaussianModel_T& rhs)
        {
            return (lhs.w / lhs.sigma) > (rhs.w / rhs.sigma);
        }

        /// Learning rate
        Parameter<double> mAlpha;
        /// A match is defined as a pixel value within mMatchThreshold standard
        /// deviations of a distribution
        Parameter<double> mMatchThreshold;
        /// Measure of the minimum portion of the data that should be accounted
        /// for by the background
        Parameter<double> mBackgroundPortion;
        /// Initial sigma (should be high), variance = sigma^2
        Parameter<double> mSigmaInit;
        /// Minimal sigma / low sigma threshold, variance = sigma^2
        Parameter<double> mSigmaMin;

        // Number of gaussian per pixel in the gaussian mixture model
        unsigned int mK;
        // Gaussian mixture model storage, for each pixel, a vector of
        // GaussianModel_T
        Matrix<std::vector<GaussianModel_T> > mModel;
        // Matching model for each pixel (int), plus flag (bool) to indicate if
        // the models for this pixel are to be updated
        Matrix<std::pair<int, bool> > mMatching;
    };
}
}

#endif // N2D2_COMPUTERVISION_GAUSSIANMIXTURE_H
