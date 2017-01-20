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

#include "Cell/FcCell_Spike.hpp"

N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Spike::mRegistrar("Spike", N2D2::FcCell_Spike::create);

N2D2::FcCell_Spike::FcCell_Spike(Network& net,
                                 const std::string& name,
                                 unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      FcCell(name, nbOutputs),
      Cell_Spike(net, name, nbOutputs),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mWeightsRelInit(this, "WeightsRelInit", 0.0, 0.05),
      mThreshold(this, "Threshold", 1.0),
      mBipolarThreshold(this, "BipolarThreshold", true),
      mLeak(this, "Leak", 0.0),
      mRefractory(this, "Refractory", 0 * TimeS),
      mTerminateDelta(this, "TerminateDelta", 0),
      mTerminateMax(this, "TerminateMax", 0)
{
    // ctor
}

void N2D2::FcCell_Spike::initialize()
{
    mSynapses.resize(
        mChannelsWidth, mChannelsHeight, mNbChannels, mOutputs.dimZ());

    for (unsigned int index = 0; index < mSynapses.size(); ++index)
        mSynapses(index) = newSynapse();

    mOutputsLastIntegration.resize(mNbOutputs, 0);
    mOutputsIntegration.resize(mNbOutputs, 0.0);
    mOutputsRefractoryEnd.resize(mNbOutputs, 0);
    mNbActivations.resize(mNbOutputs, 0);
}

void N2D2::FcCell_Spike::propagateSpike(NodeIn* origin,
                                        Time_T timestamp,
                                        EventType_T type)
{
    const Area& area = origin->getArea();

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        const Time_T delay = static_cast<Synapse_Static*>(
            mSynapses(area.x, area.y, origin->getChannel(), output))->delay;

        if (delay > 0)
            mNet.newEvent(origin, NULL, timestamp + delay, maps(output, type));
        else
            incomingSpike(origin, timestamp, maps(output, type));
    }
}

void N2D2::FcCell_Spike::incomingSpike(NodeIn* origin,
                                       Time_T timestamp,
                                       EventType_T type)
{
    // Input node coordinates
    const Area& area = origin->getArea();

    // Output node coordinates
    unsigned int output;
    bool negative;
    std::tie(output, negative) = unmaps(type);

    // Neuron state variables
    Time_T& lastIntegration = mOutputsLastIntegration[output];
    double& integration = mOutputsIntegration[output];
    Time_T& refractoryEnd = mOutputsRefractoryEnd[output];

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
        mSynapses(area.x, area.y, origin->getChannel(), output));
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

        mOutputs(output)->incomingSpike(NULL, timestamp + 1 * TimeFs, negSpike);

        if (negSpike)
            --mNbActivations[output];
        else
            ++mNbActivations[output];

        if (mTerminateDelta > 0 || mTerminateMax > 0) {
            std::vector<int> nbActivations(mNbActivations);
            std::partial_sort(nbActivations.begin(),
                              nbActivations.begin() + 2,
                              nbActivations.end(),
                              std::greater<int>());

            if ((mTerminateDelta > 0 && (nbActivations[0] - nbActivations[1]
                                         >= (int)mTerminateDelta))
                || (mTerminateMax > 0 && nbActivations[0]
                                         >= (int)mTerminateMax)) {
                mNet.stop(timestamp + 2 * TimeFs, true);
            }
        }
    }
}

void N2D2::FcCell_Spike::notify(Time_T timestamp, NotifyType notify)
{
    if (notify == Initialize) {
        if (mThreshold <= 0.0)
            throw std::domain_error("mThreshold is <= 0.0");
    } else if (notify == Reset) {
        mOutputsLastIntegration.assign(mNbOutputs, timestamp);
        mOutputsIntegration.assign(mNbOutputs, 0.0);
        mOutputsRefractoryEnd.assign(mNbOutputs, 0);
        mNbActivations.assign(mNbOutputs, 0);
    } else if (notify == Load)
        load(mNet.getLoadSavePath());
    else if (notify == Save)
        save(mNet.getLoadSavePath());
}

N2D2::NodeId_T N2D2::FcCell_Spike::getBestResponseId(bool report) const
{
    int bestScore = std::numeric_limits<int>::min();
    NodeId_T bestId = 0;

    for (std::vector<NodeOut*>::const_iterator it = mOutputs.begin(),
                                               itEnd = mOutputs.end();
         it != itEnd;
         ++it) {
        const int score = (int)(*it)->getActivity(0, 0, 0)
                          - (int)(*it)->getActivity(0, 0, 1);

        if (score > bestScore) {
            bestScore = score;
            bestId = (*it)->getId();
        }
    }

    if (report) {
        for (std::vector<NodeOut*>::const_iterator it = mOutputs.begin(),
                                                   itEnd = mOutputs.end();
             it != itEnd;
             ++it) {
            const int score = (int)(*it)->getActivity(0, 0, 0)
                              - (int)(*it)->getActivity(0, 0, 1);

            std::cout << "out #" << (*it)->getId() << ": " << score
                      << " (= " << (*it)->getActivity(0, 0, 0) << " - "
                      << (*it)->getActivity(0, 0, 1) << ")";

            if (bestId == (*it)->getId())
                std::cout << " ***";

            std::cout << std::endl;
        }
    }

    return bestId;
}

void N2D2::FcCell_Spike::saveFreeParameters(const std::string& fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    for (std::vector<Synapse*>::const_iterator it = mSynapses.begin(),
                                               itEnd = mSynapses.end();
         it != itEnd;
         ++it)
        (*it)->saveInternal(syn);
}

void N2D2::FcCell_Spike::loadFreeParameters(const std::string& fileName,
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

    for (std::vector<Synapse*>::const_iterator it = mSynapses.begin(),
                                               itEnd = mSynapses.end();
         it != itEnd;
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

N2D2::Synapse::Stats N2D2::FcCell_Spike::logStats(const std::string
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

    const unsigned int channelsSize = getNbChannels() * getChannelsWidth()
                                      * getChannelsHeight();

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        globalData << "[Output #" << output << "]\n";
        std::unique_ptr<Synapse::Stats> statsOutput(dummy->newStats());

        std::ostringstream fileName;
        fileName << dirName << "/cell-" << output << ".log";

        std::ofstream data(fileName.str().c_str());

        if (!data.good())
            throw std::runtime_error("Could not create stats log file: "
                                     + fileName.str());

        for (unsigned int i = 0; i < channelsSize; ++i) {
            std::ostringstream suffixStr;
            suffixStr << i;
            mSynapses(i, output)->logStats(data, suffixStr.str());

            mSynapses(i, output)->getStats(stats.get());
            mSynapses(i, output)->getStats(statsOutput.get());
        }

        dummy->logStats(globalData, statsOutput.get());
        globalData << "\n";
    }

    globalData << "------------------------------------------------------------"
                  "--------------------\n\n"
                  "[Global stats]\n"
                  "Cell outputs: " << getNbOutputs()
               << "\n"
                  "Cell synapses: " << getNbSynapses()
               << "\n"
                  "Cell synapses per output: "
               << getNbSynapses() / (double)getNbOutputs() << "\n";
    dummy->logStats(globalData, stats.get());
    globalData << "\n";

    Synapse::Stats globalStats = *stats.get();
    return globalStats;
}

N2D2::FcCell_Spike::~FcCell_Spike()
{
    // dtor
    std::for_each(mSynapses.begin(), mSynapses.end(), Utils::Delete());
}

N2D2::Synapse* N2D2::FcCell_Spike::newSynapse() const
{
    return new Synapse_Static(true,
                              mIncomingDelay.spreadNormal(0),
                              mWeightsRelInit.spreadNormal(-1.0, 1.0));
}
