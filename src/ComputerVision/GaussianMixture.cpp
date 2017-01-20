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

#include "ComputerVision/GaussianMixture.hpp"

N2D2::ComputerVision::GaussianMixture::GaussianMixture(unsigned int k)
    : mAlpha(this, "Alpha", 1.0e-3),
      mMatchThreshold(this, "MatchThreshold", 2.5),
      mBackgroundPortion(this, "BackgroundPortion", 0.5),
      mSigmaInit(this, "SigmaInit", 0.12),
      mSigmaMin(this, "SigmaMin", 0.075),
      mK(k)
{
    // ctor
}

N2D2::Matrix<unsigned char>
N2D2::ComputerVision::GaussianMixture::getForeground(const Matrix
                                                     <double>& frame)
{
    const unsigned int width = frame.cols();
    const unsigned int height = frame.rows();

    if (mModel.empty()) {
        // Initialize the model
        std::vector<GaussianModel_T> init;
        init.reserve(mK);

        for (unsigned int i = 0; i < mK; ++i)
            init.push_back(GaussianModel_T(
                1.0 / mK, (i + 1) / (double)(mK + 1), (double)mSigmaInit));

        mModel.resize(height, width, init);
        mMatching.resize(height, width);
    } else if (mModel.cols() != width || mModel.rows() != height)
        throw std::runtime_error("Frame size does not match model size");

    Matrix<unsigned char> foreground(height, width, 0);

    const unsigned int size = frame.size();

#pragma omp parallel for if (size > 16)
    for (int index = 0; index < (int)size; ++index) {
        // For each pixel
        int matchingModel = -1;

        // The first B distributions are chosen as the background model
        double wSum = 0.0;
        int backgroundLimit = -1;

        assert(mModel(index).size() == mK);

        for (unsigned int model = 0; model < mK; ++model) {
            // For each gaussian
            if (matchingModel < 0
                && std::fabs(frame(index) - mModel(index)[model].mu)
                   < mMatchThreshold * mModel(index)[model].sigma) {
                // A match is found
                matchingModel = model;
            }

            wSum += mModel(index)[model].w;

            if (backgroundLimit < 0 && wSum > mBackgroundPortion)
                backgroundLimit = model;
        }

        if (matchingModel > backgroundLimit)
            foreground(index) = 1;

        mMatching(index) = std::make_pair(matchingModel, false);
    }

    return foreground;
}

void N2D2::ComputerVision::GaussianMixture::excludeRoi(const ROI::Roi_T& roi)
{
    for (unsigned int i = roi.i0; i <= roi.i1; ++i) {
        for (unsigned int j = roi.j0; j <= roi.j1; ++j)
            mMatching(i, j).second = true;
    }
}

void N2D2::ComputerVision::GaussianMixture::updateModel(const Matrix
                                                        <double>& frame)
{
    const unsigned int size = frame.size();

#pragma omp parallel for if (size > 16)
    for (int index = 0; index < (int)size; ++index) {
        if (mMatching(index).second)
            continue;

        double wSum = 0.0;

        if (mMatching(index).first >= 0) {
            // Weights adjustment
            for (unsigned int model = 0; model < mK; ++model) {
                GaussianModel_T& pm = mModel(index)[model];

                // For each gaussian
                if ((int)model == mMatching(index).first) {
                    // Update the gaussian model
                    const double x = frame(index) - pm.mu;
                    const double sigma2 = pm.sigma * pm.sigma;
                    const double rho
                        = mAlpha * (1.0 / (std::sqrt(2.0 * M_PI) * pm.sigma))
                          * std::exp(-(x * x) / (2.0 * sigma2));

                    pm.w = (1.0 - mAlpha) * pm.w + mAlpha;
                    pm.mu = (1.0 - rho) * pm.mu + rho * frame(index);

                    const double newX = frame(index) - pm.mu;

                    pm.sigma = std::max(
                        (double)mSigmaMin,
                        std::sqrt((1.0 - rho) * sigma2 + rho * newX * newX));
                } else
                    pm.w = (1.0 - mAlpha) * pm.w;

                wSum += pm.w;
            }
        } else {
            GaussianModel_T& pm = mModel(index).back();

            wSum = 1.0 - pm.w;

            // Replace the least probable distribution
            pm.w = mAlpha; // low prior weight
            pm.mu = frame(index); // current value as its mean value
            pm.sigma = mSigmaInit; // initially high variance

            wSum += pm.w;
        }

        // Weights re-normalization
        for (unsigned int model = 0; model < mK; ++model)
            mModel(index)[model].w /= wSum;

        // The Gaussians are ordered by the value of w/sigma
        std::sort(mModel(index).begin(), mModel(index).end(), modelCompare);
    }
}

N2D2::Matrix<double> N2D2::ComputerVision::GaussianMixture::getBaseBackground(
    unsigned int level) const
{
    if (level >= mK)
        throw std::out_of_range("Background level is out of range");

    Matrix<double> background(mModel.rows(), mModel.cols());

    for (unsigned int index = 0, size = mModel.size(); index < size; ++index)
        background(index) = mModel(index)[level].mu;

    return background;
}

void N2D2::ComputerVision::GaussianMixture::getPixelModel(unsigned int x,
                                                          unsigned int y) const
{
    static Gnuplot gnuplot;
    gnuplot.setXlabel("Value");

    std::ostringstream label;
    label << "\"@ (" << x << ", " << y << ")\" at graph 0.1, graph 0.9 front";

    gnuplot.unset("label");
    gnuplot.set("label", label.str());
    gnuplot.showOnScreen();

    std::stringstream plotCmd;

    for (unsigned int model = 0; model < mK; ++model) {
        assert(mModel(y, x).size() == mK);

        if (model > 0)
            plotCmd << ", \"\" ";

        plotCmd << "using 1:" << (model + 2) << " with lines title \"" << model
                << "\"";
    }

    gnuplot.plot("-", plotCmd.str());

    for (double i = 0.0; i < 1.0; i += 0.001) {
        std::stringstream cmd;
        cmd << i;

        for (unsigned int model = 0; model < mK; ++model) {
            const GaussianModel_T& pm = mModel(y, x)[model];
            const double v = pm.w * (1.0 / (std::sqrt(2.0 * M_PI) * pm.sigma))
                             * std::exp(-(i - pm.mu) * (i - pm.mu)
                                        / (2.0 * pm.sigma * pm.sigma));

            cmd << " " << v;
        }

        gnuplot << cmd.str();
    }

    gnuplot << "e";
}

void N2D2::ComputerVision::GaussianMixture::load(const std::string& fileName,
                                                 bool ignoreNotExists)
{
    std::ifstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good()) {
        if (ignoreNotExists) {
            std::cout << "Notice: Could not open data file: " << fileName
                      << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open data file: " + fileName);
    }

    unsigned int width, height;
    data.read(reinterpret_cast<char*>(&width), sizeof(width));
    data.read(reinterpret_cast<char*>(&height), sizeof(height));
    data.read(reinterpret_cast<char*>(&mK), sizeof(mK));

    if (!data.good())
        throw std::runtime_error("Error while reading data file: " + fileName);

    // Initialize the model
    mModel.resize(height, width);
    mMatching.resize(height, width);

    for (unsigned int index = 0, size = mModel.size(); index < size; ++index) {
        mModel(index).resize(mK, GaussianModel_T(0.0, 0.0, 0.0));

        data.read(reinterpret_cast<char*>(&mModel(index)[0]),
                  mModel(index).size() * sizeof(GaussianModel_T));
    }

    if (data.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in data file: " + fileName);
    else if (!data.good())
        throw std::runtime_error("Error while reading data file: " + fileName);
    else if (data.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Data file size larger than expected: "
                                 + fileName);
}

void N2D2::ComputerVision::GaussianMixture::save(const std::string
                                                 & fileName) const
{
    if (mModel.empty())
        throw std::runtime_error("No model to save: model is empty");

    std::ofstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not create data file: " + fileName);

    const unsigned int width = mModel.cols();
    const unsigned int height = mModel.rows();

    data.write(reinterpret_cast<const char*>(&width), sizeof(width));
    data.write(reinterpret_cast<const char*>(&height), sizeof(height));
    data.write(reinterpret_cast<const char*>(&mK), sizeof(mK));

    for (unsigned int index = 0, size = mModel.size(); index < size; ++index) {
        assert(mModel(index).size() == mK);

        data.write(reinterpret_cast<const char*>(&mModel(index)[0]),
                   mModel(index).size() * sizeof(GaussianModel_T));
    }

    if (!data.good())
        throw std::runtime_error(
            "GaussianMixture::save(): error writing data file" + fileName);
}
