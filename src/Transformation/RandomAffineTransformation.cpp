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

#include "Transformation/RandomAffineTransformation.hpp"

const char* N2D2::RandomAffineTransformation::Type = "RandomAffine";

N2D2::RandomAffineTransformation::RandomAffineTransformation(double gainVar,
                                                             double biasVar)
    : mGainRange(std::vector<std::pair<double, double> >({
        std::pair<double, double>(1.0 - gainVar, 1.0 + gainVar)
      })),
      mBiasRange(std::vector<std::pair<double, double> >({
        std::pair<double, double>(-biasVar, biasVar)
      })),
      mDisjointGamma(this, "DisjointGamma", false),
      mChannelsMask(this, "ChannelsMask", std::vector<bool>())
{
    // ctor
}

N2D2::RandomAffineTransformation::RandomAffineTransformation(
    const std::vector<std::pair<double, double> >& gainRange,
    const std::vector<std::pair<double, double> >& biasRange,
    const std::vector<std::pair<double, double> >& gammaRange,
    const std::vector<double>& gainVarProb,
    const std::vector<double>& biasVarProb,
    const std::vector<double>& gammaVarProb)
    : mGainRange(gainRange),
      mBiasRange(biasRange),
      mGammaRange(gammaRange),
      mGainVarProb(gainVarProb),
      mBiasVarProb(biasVarProb),
      mGammaVarProb(gammaVarProb),
      mDisjointGamma(this, "DisjointGamma", false),
      mChannelsMask(this, "ChannelsMask", std::vector<bool>())
{
    // ctor
}

N2D2::RandomAffineTransformation::RandomAffineTransformation(
    const RandomAffineTransformation& trans)
    : mGainRange(trans.mGainRange),
      mBiasRange(trans.mBiasRange),
      mGammaRange(trans.mGammaRange),
      mGainVarProb(trans.mGainVarProb),
      mBiasVarProb(trans.mBiasVarProb),
      mGammaVarProb(trans.mGammaVarProb),
      mDisjointGamma(this, "DisjointGamma", trans.mDisjointGamma),
      mChannelsMask(this, "ChannelsMask", trans.mChannelsMask)
{
    // copy-ctor
}

void
N2D2::RandomAffineTransformation::apply(cv::Mat& frame,
                                        cv::Mat& /*labels*/,
                                        std::vector
                                        <std::shared_ptr<ROI> >& /*labelsROI*/,
                                        int /*id*/)
{
    const int nbChannels = frame.channels();

    const bool isGlobalGainVar = (mGainVarProb.size() == 1) ?
        Random::randBernoulli(mGainVarProb[0]) : false;
    const bool isGlobalBiasVar = (mBiasVarProb.size() == 1) ?
        Random::randBernoulli(mBiasVarProb[0]) : false;
    const bool isGlobalGammaVar = (mGammaVarProb.size() == 1) ?
        Random::randBernoulli(mGammaVarProb[0]) : false;

    const double globalGain = (mGainRange.size() == 1
                               && mGainVarProb.size() <= 1) ?
        Random::randUniform(mGainRange[0].first, mGainRange[0].second) : 1.0;
    const double globalBias = (mBiasRange.size() == 1
                               && mBiasVarProb.size() <= 1) ?
        Random::randUniform(mBiasRange[0].first, mBiasRange[0].second) : 0.0;
    const double globalGamma = (mGammaRange.size() == 1
                                && mGammaVarProb.size() <= 1) ?
        Random::randUniform(mGammaRange[0].first, mGammaRange[0].second) : 1.0;

    const bool reqSplit = (mGainVarProb.size() > 1 || mGainRange.size() > 1
                        || mBiasVarProb.size() > 1 || mBiasRange.size() > 1
                        || mGammaVarProb.size() > 1 || mGammaRange.size() > 1);

    std::vector<cv::Mat> channels;

    if (!mChannelsMask->empty() || reqSplit)
        cv::split(frame, channels);
    else
        channels.push_back(frame.reshape(1));

    for (std::vector<cv::Mat>::iterator it = channels.begin(),
        itBegin = channels.begin(), itEnd = channels.end(); it != itEnd; ++it)
    {
        const unsigned int ch = it - itBegin;

        if (!mChannelsMask->empty()
            && (ch >= mChannelsMask->size()
                || !(*(mChannelsMask->begin() + ch))))
        {
            // This channel is masked, no transformation applied
            continue;
        }

        bool isGainVar =
            // A single prob means the same sampling for every channels
            (mGainVarProb.size() == 1) ? isGlobalGainVar :
            // A prob is specified for this channel => separate sampling
            (ch < mGainVarProb.size()) ? Random::randBernoulli(mGainVarProb[ch])
            // By default, if no prob specified, apply a gain variation
                                       : true;
        bool isBiasVar =
            (mBiasVarProb.size() == 1) ? isGlobalBiasVar :
            (ch < mBiasVarProb.size()) ? Random::randBernoulli(mBiasVarProb[ch])
                                       : true;
        const bool isGammaVar =
            (mGammaVarProb.size() == 1) ? isGlobalGammaVar :
            (ch < mGammaVarProb.size()) ? Random::randBernoulli(mGammaVarProb[ch])
                                       : true;

        if (isGammaVar && mDisjointGamma) {
            isGainVar = false;
            isBiasVar = false;
        }

        double gain = 1.0;

        if (isGainVar) {
            gain =
                // A single range + single/no prob means the same sampling for every channels
                (mGainRange.size() == 1 && mGainVarProb.size() <= 1) ? globalGain :
                // A single range is specified but with different probs => separate sampling
                (mGainRange.size() == 1) ? Random::randUniform(mGainRange[0].first,
                                                               mGainRange[0].second) :
                // A range is specified for this channel => separate sampling
                (ch < mGainRange.size()) ? Random::randUniform(mGainRange[ch].first,
                                                               mGainRange[ch].second)
                // By default, if no range specified, no gain variation
                                           : 1.0;
        }

        double bias = 0.0;

        if (isBiasVar) {
            bias =
                (mBiasRange.size() == 1 && mBiasVarProb.size() <= 1) ? globalBias :
                (mBiasRange.size() == 1) ? Random::randUniform(mBiasRange[0].first,
                                                               mBiasRange[0].second) :
                (ch < mBiasRange.size()) ? Random::randUniform(mBiasRange[ch].first,
                                                               mBiasRange[ch].second)
                                           : 0.0;
        }

        double gamma = 1.0;

        if (isGammaVar) {
            gamma =
                (mGammaRange.size() == 1 && mGammaVarProb.size() <= 1) ? globalGamma :
                (mGammaRange.size() == 1) ? Random::randUniform(mGammaRange[0].first,
                                                               mGammaRange[0].second) :
                (ch < mGammaRange.size()) ? Random::randUniform(mGammaRange[ch].first,
                                                               mGammaRange[ch].second)
                                           : 1.0;
        }

        switch ((*it).depth()) {
        case CV_8U:
            applyRandomAffine<unsigned char>((*it), gain, bias, gamma);
            break;
        case CV_8S:
            applyRandomAffine<char>((*it), gain, bias, gamma);
            break;
        case CV_16U:
            applyRandomAffine<unsigned short>((*it), gain, bias, gamma);
            break;
        case CV_16S:
            applyRandomAffine<short>((*it), gain, bias, gamma);
            break;
        case CV_32S:
            applyRandomAffine<int>((*it), gain, bias, gamma);
            break;
        case CV_32F:
            applyRandomAffine<float>((*it), gain, bias, gamma);
            break;
        case CV_64F:
            applyRandomAffine<double>((*it), gain, bias, gamma);
            break;
        default:
            throw std::runtime_error(
                "Cannot apply affine transformation: incompatible type.");
        }
    }

    if (!mChannelsMask->empty() || reqSplit)
        cv::merge(channels, frame);
    else
        frame = channels[0].reshape(nbChannels);
}

N2D2::RandomAffineTransformation::~RandomAffineTransformation() {
    
}
