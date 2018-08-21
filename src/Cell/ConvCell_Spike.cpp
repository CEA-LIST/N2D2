/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "Cell/ConvCell_Spike.hpp"

N2D2::Registrar<N2D2::ConvCell>
N2D2::ConvCell_Spike::mRegistrar("Spike",
    N2D2::ConvCell_Spike::create,
    N2D2::Registrar<N2D2::ConvCell>::Type<Float_T>());

N2D2::ConvCell_Spike::ConvCell_Spike(Network& net,
                                 const std::string& name,
                                 const std::vector<unsigned int>& kernelDims,
                                 unsigned int nbOutputs,
                                 const std::vector<unsigned int>& subSampleDims,
                                 const std::vector<unsigned int>& strideDims,
                                 const std::vector<int>& paddingDims)
    : Cell(name, nbOutputs),
      ConvCell(name,
               kernelDims,
               nbOutputs,
               subSampleDims,
               strideDims,
               paddingDims),
      Cell_Spike(net, name, nbOutputs),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mWeightsRelInit(this, "WeightsRelInit", 0.0, 0.05),
      mThreshold(this, "Threshold", 1.0),
      mBipolarThreshold(this, "BipolarThreshold", true),
      mLeak(this, "Leak", 0.0),
      mRefractory(this, "Refractory", 0 * TimeS)
{
    // ctor
    if (kernelDims.size() != 2) {
        throw std::domain_error("ConvCell_Spike: only 2D convolution is"
                                " supported");
    }

    if (subSampleDims.size() != kernelDims.size()) {
        throw std::domain_error("ConvCell_Spike: the number of dimensions of"
                                " subSample must match the number of dimensions"
                                " of the kernel.");
    }

    if (strideDims.size() != kernelDims.size()) {
        throw std::domain_error("ConvCell_Spike: the number of dimensions of"
                                " stride must match the number of dimensions"
                                " of the kernel.");
    }

    if (paddingDims.size() != kernelDims.size()) {
        throw std::domain_error("ConvCell_Spike: the number of dimensions of"
                                " passing must match the number of dimensions"
                                " of the kernel.");
    }

    mWeightsFiller = std::make_shared<NormalFiller<Float_T> >(0.0, 0.05);
    mBiasFiller = std::make_shared<NormalFiller<Float_T> >(0.0, 0.05);
}

void N2D2::ConvCell_Spike::initialize()
{
    mSharedSynapses.resize(
        {mKernelDims[0], mKernelDims[1], getNbChannels(), getNbOutputs()});

    for (unsigned int index = 0, size = mSharedSynapses.size(); index < size;
         ++index)
        mSharedSynapses(index) = newSynapse();

    mOutputsLastIntegration.resize(
        {mOutputsDims[0], mOutputsDims[1], getNbOutputs(), 1}, 0);
    mOutputsIntegration.resize(
        {mOutputsDims[0], mOutputsDims[1], getNbOutputs(), 1}, 0.0);
    mOutputsRefractoryEnd.resize(
        {mOutputsDims[0], mOutputsDims[1], getNbOutputs(), 1}, 0);
}

void N2D2::ConvCell_Spike::propagateSpike(NodeIn* origin,
                                          Time_T timestamp,
                                          EventType_T type)
{
    const Area& area = origin->getArea();
    const unsigned int oxStride
        = mStrideDims[0]
          * (unsigned int)((mInputsDims[0] + 2 * mPaddingDims[0] - mKernelDims[0]
                            + mStrideDims[0]) / (double)mStrideDims[0]);
    const unsigned int oyStride
        = mStrideDims[1]
          * (unsigned int)((mInputsDims[1] + 2 * mPaddingDims[1] - mKernelDims[1]
                            + mStrideDims[1]) / (double)mStrideDims[1]);
    const unsigned int ixPad = area.x + mPaddingDims[0];
    const unsigned int iyPad = area.y + mPaddingDims[1];
    const unsigned int sxMax = std::min(mKernelDims[0], ixPad + 1);
    const unsigned int syMax = std::min(mKernelDims[1], iyPad + 1);

    for (unsigned int sy = iyPad % mStrideDims[1], sx0 = ixPad % mStrideDims[0]; sy < syMax;
         sy += mStrideDims[1]) {
        if (iyPad >= oyStride + sy)
            continue;

        for (unsigned int sx = sx0; sx < sxMax; sx += mStrideDims[0]) {
            // Border conditions
            if (ixPad >= oxStride + sx)
                continue;

            // Output node coordinates
            const unsigned int ox = (ixPad - sx) / mStrideDims[0];
            const unsigned int oy = (iyPad - sy) / mStrideDims[1];

            for (unsigned int output = 0; output < getNbOutputs(); ++output) {
                if (!isConnection(origin->getChannel(), output))
                    continue;

                const Time_T delay = static_cast<Synapse_Static*>(
                    mSharedSynapses(sx, sy, origin->getChannel(), output))
                                         ->delay;

                if (delay > 0)
                    mNet.newEvent(origin,
                                  NULL,
                                  timestamp + delay,
                                  maps(output, ox, oy, type));
                else
                    incomingSpike(
                        origin, timestamp, maps(output, ox, oy, type));
            }
        }
    }
}

void N2D2::ConvCell_Spike::incomingSpike(NodeIn* origin,
                                         Time_T timestamp,
                                         EventType_T type)
{
    // Input node coordinates
    const Area& area = origin->getArea();

    // Output node coordinates
    unsigned int output, ox, oy;
    bool negative;
    std::tie(output, ox, oy, negative) = unmaps(type);

    const unsigned int subOx = ox / mSubSampleDims[0];
    const unsigned int subOy = oy / mSubSampleDims[1];

    // Synapse coordinates
    const unsigned int synX = area.x - ox * mStrideDims[0] + mPaddingDims[0];
    const unsigned int synY = area.y - oy * mStrideDims[1] + mPaddingDims[1];

    // Neuron state variables
    Time_T& lastIntegration = mOutputsLastIntegration(subOx, subOy, output, 0);
    double& integration = mOutputsIntegration(subOx, subOy, output, 0);
    Time_T& refractoryEnd = mOutputsRefractoryEnd(subOx, subOy, output, 0);

    // Integrates
    if (mLeak > 0.0) {
        const Time_T dt = timestamp - lastIntegration;
        const double expVal = -((double)dt) / ((double)mLeak);

        if (expVal > std::log(1e-20))
            integration *= std::exp(expVal);
        else {
            integration = 0.0;
            // std::cout << "Notice: integration leaked to 0 (no activity during
            // " << dt/((double) TimeS) << " s = "
            //    << (-expVal) << " * mLeak)." << std::endl;
        }
    }

    lastIntegration = timestamp;

    Synapse_Static* synapse = static_cast<Synapse_Static*>(
        mSharedSynapses(synX, synY, origin->getChannel(), output));
    integration += (negative) ? -synapse->weight : synapse->weight;

    // Stats
    ++synapse->statsReadEvents;

    if ((integration >= mThreshold
         || (mBipolarThreshold && (-integration) >= mThreshold))
        && timestamp >= refractoryEnd) {
        const bool negSpike = (integration < 0);

        refractoryEnd = timestamp + mRefractory;

        // If the integration is reset to 0, part of the contribution of the
        // current spike is lost.
        // Performances are significantly better (~0.8% on GTSRB) if the value
        // above the threshold is kept.
        if (negSpike)
            integration += mThreshold;
        else
            integration -= mThreshold;

        mOutputs(subOx, subOy, output, 0)
            ->incomingSpike(NULL, timestamp + 1 * TimeFs, negSpike);
    }
}

void N2D2::ConvCell_Spike::notify(Time_T timestamp, NotifyType notify)
{
    if (notify == Initialize) {
        if (mThreshold <= 0.0)
            throw std::domain_error("mThreshold is <= 0.0");
    } else if (notify == Reset) {
        mOutputsLastIntegration.assign(
            {mOutputsDims[0], mOutputsDims[1], getNbOutputs(), 1}, timestamp);
        mOutputsIntegration.assign(
            {mOutputsDims[0], mOutputsDims[1], getNbOutputs(), 1}, 0.0);
        mOutputsRefractoryEnd.assign(
            {mOutputsDims[0], mOutputsDims[1], getNbOutputs(), 1}, 0);
    } else if (notify == Load)
        load(mNet.getLoadSavePath());
    else if (notify == Save)
        save(mNet.getLoadSavePath());
}

cv::Mat N2D2::ConvCell_Spike::reconstructActivity(unsigned int output,
                                                  Time_T start,
                                                  Time_T stop,
                                                  bool normalize) const
{
    if (output >= getNbOutputs())
        throw std::domain_error(
            "ConvCell_Spike::reconstructActivity(): output not within range.");

    unsigned int maxValue = 0;

    for (Matrix<NodeOut*>::const_iterator it = mOutputs.begin(),
                                          itEnd = mOutputs.end();
         it != itEnd;
         ++it)
        maxValue = std::max(maxValue, (*it)->getActivity(start, stop));

    cv::Mat img(cv::Size(mOutputsDims[0], mOutputsDims[1]), CV_8UC1, 0.0);

    if (maxValue > 0) {
        for (unsigned int y = 0; y < mOutputsDims[1]; ++y) {
            for (unsigned int x = 0; x < mOutputsDims[0]; ++x)
                img.at<unsigned char>(y, x)
                    = 255.0 * (mOutputs(x, y, output, 0)->getActivity(
                                   start, stop) / (double)maxValue);
        }
    }

    if (normalize) {
        cv::Mat imgNorm;
        cv::normalize(img, imgNorm, 0.0, 255.0, cv::NORM_MINMAX);
        img = imgNorm;
    }

    return img;
}

void N2D2::ConvCell_Spike::reconstructActivities(const std::string& dirName,
                                                 Time_T start,
                                                 Time_T stop,
                                                 bool normalize) const
{
    Utils::createDirectories(dirName);

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        std::ostringstream fileName;
        fileName << dirName << "/cell-activity-" << output << ".jpg";

        cv::Mat img;
        cv::resize(reconstructActivity(output, start, stop, normalize),
                   img,
                   cv::Size(512, 512),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);

        if (!cv::imwrite(fileName.str(), img))
            throw std::runtime_error("Unable to write image: "
                                     + fileName.str());
    }
}

void N2D2::ConvCell_Spike::saveFreeParameters(const std::string& fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    for (std::vector<Synapse*>::const_iterator it = mSharedSynapses.begin();
         it != mSharedSynapses.end();
         ++it)
        (*it)->saveInternal(syn);
}

void N2D2::ConvCell_Spike::loadFreeParameters(const std::string& fileName,
                                              bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file (.SYN): "
                                     + fileName);
    }

    for (std::vector<Synapse*>::iterator it = mSharedSynapses.begin();
         it != mSharedSynapses.end();
         ++it)
        (*it)->loadInternal(syn);

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in synaptic file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading synaptic file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

N2D2::Synapse::Stats N2D2::ConvCell_Spike::logStats(const std::string
                                                    & dirName) const
{
    Utils::createDirectories(dirName);

    std::unique_ptr<Synapse> dummy(newSynapse());
    std::unique_ptr<Synapse::Stats> stats(dummy->newStats());

    std::ofstream globalData((dirName + ".log").c_str());

    if (!globalData.good())
        throw std::runtime_error("Could not create stats log file: "
                                 + (dirName + ".log"));

    globalData.imbue(Utils::locale);

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        globalData << "[Output #" << output << "]\n";
        std::unique_ptr<Synapse::Stats> statsOutput(dummy->newStats());

        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            std::ostringstream fileName;
            fileName << dirName << "/cell-" << output << "-[" << channel
                     << "].log";

            std::ofstream data(fileName.str().c_str());

            if (!data.good())
                throw std::runtime_error("Could not create stats log file: "
                                         + fileName.str());

            globalData << "[Channel #" << channel << "]\n";
            std::unique_ptr<Synapse::Stats> statsKernel(dummy->newStats());

            for (unsigned int x = 0; x < mKernelDims[0]; ++x) {
                for (unsigned int y = 0; y < mKernelDims[1]; ++y) {
                    std::ostringstream suffixStr;
                    suffixStr << x << " " << y;
                    mSharedSynapses(x, y, channel, output)
                        ->logStats(data, suffixStr.str());

                    mSharedSynapses(x, y, channel, output)
                        ->getStats(stats.get());
                    mSharedSynapses(x, y, channel, output)
                        ->getStats(statsOutput.get());
                    mSharedSynapses(x, y, channel, output)
                        ->getStats(statsKernel.get());
                }
            }

            dummy->logStats(globalData, statsKernel.get());
            globalData << "\n";
        }

        if (getNbChannels() > 1) {
            dummy->logStats(globalData, statsOutput.get());
            globalData << "\n";
        }
    }

    const unsigned int nbOutputNodes = getNbOutputs() * getOutputsWidth()
                                       * getOutputsHeight();

    globalData << "------------------------------------------------------------"
                  "--------------------\n\n"
                  "[Global stats]\n"
                  "Cell outputs: " << getNbOutputs() << "\n"
                                                        "Cell outputs size: "
               << getOutputsWidth() * getOutputsHeight() << " ("
               << getOutputsWidth() << "x" << getOutputsHeight()
               << ")\n"
                  "Cell output nodes: " << nbOutputNodes
               << "\n"
                  "Cell shared synapses: " << getNbSharedSynapses()
               << "\n"
                  "Cell shared synapses per output: "
               << getNbSharedSynapses() / (double)getNbOutputs()
               << "\n"
                  "Cell shared synapses per output node: "
               << getNbSharedSynapses() / (double)nbOutputNodes
               << "\n"
                  "Cell virtual synapses: " << getNbVirtualSynapses()
               << "\n"
                  "Cell virtual synapses per output: "
               << getNbVirtualSynapses() / (double)getNbOutputs()
               << "\n"
                  "Cell virtual synapses per output node: "
               << getNbVirtualSynapses() / (double)nbOutputNodes << "\n";
    dummy->logStats(globalData, stats.get());
    globalData << "\n";

    Synapse::Stats globalStats = *stats.get();
    return globalStats;
}

N2D2::Synapse* N2D2::ConvCell_Spike::newSynapse() const
{
    return new Synapse_Static(true,
                              mIncomingDelay.spreadNormal(0),
                              mWeightsRelInit.spreadNormal(-1.0, 1.0));
}

N2D2::ConvCell_Spike::~ConvCell_Spike()
{
    // dtor
    std::for_each(
        mSharedSynapses.begin(), mSharedSynapses.end(), Utils::Delete());
}

void N2D2::addInput(Xcell& cell,
                    ConvCell_Spike& convCell,
                    unsigned int output,
                    unsigned int x0,
                    unsigned int y0,
                    unsigned int width,
                    unsigned int height)
{
    if (width == 0)
        width = convCell.getOutputsWidth() - x0;
    if (height == 0)
        height = convCell.getOutputsHeight() - y0;

    for (unsigned int x = x0; x < x0 + width; ++x) {
        for (unsigned int y = y0; y < y0 + height; ++y)
            cell.addInput(convCell.getOutput(output, x, y));
    }
}

void N2D2::addInput(Xcell& cell, ConvCell_Spike& convCell)
{
    for (unsigned int output = 0, nbOuputs = convCell.getNbOutputs();
         output < nbOuputs;
         ++output)
        addInput(cell, convCell, output);
}
