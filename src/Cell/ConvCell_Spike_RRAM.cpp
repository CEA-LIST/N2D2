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

#include "Cell/ConvCell_Spike_RRAM.hpp"
#include "Cell/NodeIn.hpp"
#include "Cell/NodeOut.hpp"

N2D2::Registrar<N2D2::ConvCell>
N2D2::ConvCell_Spike_RRAM::mRegistrar("Spike_RRAM",
    N2D2::ConvCell_Spike_RRAM::create,
    N2D2::Registrar<N2D2::ConvCell>::Type<Float_T>());

N2D2::ConvCell_Spike_RRAM::ConvCell_Spike_RRAM(Network& net,
                                 const std::string& name,
                                 const std::vector<unsigned int>& kernelDims,
                                 unsigned int nbOutputs,
                                 const std::vector<unsigned int>& subSampleDims,
                                 const std::vector<unsigned int>& strideDims,
                                 const std::vector<int>& paddingDims,
                                 const std::vector<unsigned int>& dilationDims)
    : Cell(name, nbOutputs),
      ConvCell(name,
               kernelDims,
               nbOutputs,
               subSampleDims,
               strideDims,
               paddingDims,
               dilationDims),
      ConvCell_Spike(net,
                     name,
                     kernelDims,
                     nbOutputs,
                     subSampleDims,
                     strideDims,
                     paddingDims,
                     dilationDims),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mWeightsMinMean(this, "WeightsMinMean", 1, 0.1),
      mWeightsMaxMean(this, "WeightsMaxMean", 100, 10.0),
      mWeightsMinVarSlope(this, "WeightsMinVarSlope", 0.0),
      mWeightsMinVarOrigin(this, "WeightsMinVarOrigin", 0.0),
      mWeightsMaxVarSlope(this, "WeightsMaxVarSlope", 0.0),
      mWeightsMaxVarOrigin(this, "WeightsMaxVarOrigin", 0.0),
      mWeightsSetProba(this, "WeightsSetProba", 1.0),
      mWeightsResetProba(this, "WeightsResetProba", 1.0),
      mSynapticRedundancy(this, "SynapticRedundancy", 1U),
      mBipolarWeights(this, "BipolarWeights", false),
      mBipolarIntegration(this, "BipolarIntegration", false),
      mLtpProba(this, "LtpProba", 0.2),
      mLtdProba(this, "LtdProba", 0.1),
      mStdpLtp(this, "StdpLtp", 1000 * TimePs),
      mInhibitRefractory(this, "InhibitRefractory", 0 * TimeS),
      mEnableStdp(this, "EnableStdp", true),
      mRefractoryIntegration(this, "RefractoryIntegration", true),
      mDigitalIntegration(this, "DigitalIntegration", false)
{
    // ctor
}

void N2D2::ConvCell_Spike_RRAM::initialize()
{
    ConvCell_Spike::initialize();

    std::vector<size_t> inputsDims = mInputsDims;
    inputsDims.push_back(1);        // batch

    mInputsActivationTime.resize(inputsDims);
}

void N2D2::ConvCell_Spike_RRAM::propagateSpike(NodeIn* origin,
                                               Time_T timestamp,
                                               EventType_T type)
{
    const Area& area = origin->getArea();
    const unsigned int channel = origin->getChannel();

    mInputsActivationTime(area.x, area.y, channel, 0) = timestamp;

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
                if (!isConnection(channel, output))
                    continue;

                const Time_T delay = static_cast<Synapse_RRAM*>(
                    mSharedSynapses(sx, sy, channel, output))->delay;

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

void N2D2::ConvCell_Spike_RRAM::incomingSpike(NodeIn* origin,
                                              Time_T timestamp,
                                              EventType_T type)
{
    // Output node coordinates
    unsigned int output, ox, oy;
    bool negative;
    std::tie(output, ox, oy, negative) = unmaps(type);

    const unsigned int subOx = ox / mSubSampleDims[0];
    const unsigned int subOy = oy / mSubSampleDims[1];

    // Synapse coordinates
    const Area& area = origin->getArea();
    const unsigned int synX = area.x - ox * mStrideDims[0] + mPaddingDims[0];
    const unsigned int synY = area.y - oy * mStrideDims[1] + mPaddingDims[1];

    // Neuron state variables
    Time_T& lastIntegration = mOutputsLastIntegration(subOx, subOy, output, 0);
    double& integration = mOutputsIntegration(subOx, subOy, output, 0);
    Time_T& refractoryEnd = mOutputsRefractoryEnd(subOx, subOy, output, 0);

    // Integrates
    if (mLeak > 0) {
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

    Synapse_RRAM* synapse = static_cast<Synapse_RRAM*>(
        mSharedSynapses(synX, synY, origin->getChannel(), output));

    if (mDigitalIntegration) {
        const double threshold
            = (mWeightsMaxMean.mean() + mWeightsMinMean.mean()) / 2.0;

        if (mBipolarIntegration && !mBipolarWeights) {
            // For off-line learning, spike-based feed-forward, without bipolar
            // synapses
            integration += (negative)
                               ? -2 * synapse->getDigitalWeight(threshold)
                                 + mSynapticRedundancy
                               : 2 * synapse->getDigitalWeight(threshold)
                                 - mSynapticRedundancy;
        } else
            integration += (negative) ? -synapse->getDigitalWeight(threshold)
                                      : synapse->getDigitalWeight(threshold);
    } else {
        if (mBipolarIntegration && !mBipolarWeights) {
            // For off-line learning, spike-based feed-forward, without bipolar
            // synapses
            integration
                += (negative)
                       ? -2.0 * synapse->getWeight()
                         + mSynapticRedundancy
                           * (mWeightsMaxMean.mean() + mWeightsMinMean.mean())
                       : 2.0 * synapse->getWeight()
                         - mSynapticRedundancy
                           * (mWeightsMaxMean.mean() + mWeightsMinMean.mean());
        } else
            integration += (negative) ? -synapse->getWeight()
                                      : synapse->getWeight();
    }

    // Stats
    ++synapse->statsReadEvents;

    for (unsigned int dev = 0, devSize = synapse->devices.size(); dev < devSize;
         ++dev)
        synapse->stats[dev].statsReadEnergy += synapse->devices[dev].weight;

    // For STDP, integration stays at 0 during refractory period
    // For off-line learning, integration must continue during refractory period
    if (!mRefractoryIntegration && timestamp < refractoryEnd)
        integration = 0.0;

    const double scaledThres = mThreshold * mSynapticRedundancy;

    if ((integration >= scaledThres
         || (mBipolarThreshold && (-integration) >= scaledThres))
        && timestamp >= refractoryEnd) {
        const bool negSpike = (integration < 0);

        if (mEnableStdp) {
            const unsigned int sxMin = (unsigned int)std::max(
                (int)mPaddingDims[0] - (int)(ox * mStrideDims[0]), 0);
            const unsigned int syMin = (unsigned int)std::max(
                (int)mPaddingDims[1] - (int)(oy * mStrideDims[1]), 0);
            const unsigned int sxMax = Utils::clamp<int>(
                mInputsDims[0] + mPaddingDims[0] - ox * mStrideDims[0], 0, mKernelDims[0]);
            const unsigned int syMax = Utils::clamp<int>(
                mInputsDims[1] + mPaddingDims[1] - oy * mStrideDims[1], 0, mKernelDims[1]);

            for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
            {
                if (!isConnection(channel, output))
                    continue;

                for (unsigned int sy = syMin; sy < syMax; ++sy) {
                    for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                        const unsigned int ix = (int)(ox * mStrideDims[0] + sx)
                                                - (int)mPaddingDims[0];
                        const unsigned int iy = (int)(oy * mStrideDims[1] + sy)
                                                - (int)mPaddingDims[1];

                        const Time_T lastSpike
                            = mInputsActivationTime(ix, iy, channel, 0);
                        Synapse_RRAM* synapse = static_cast<Synapse_RRAM*>(
                            mSharedSynapses(sx, sy, channel, output));

                        if (lastSpike > 0 && lastSpike + mStdpLtp >= timestamp)
                            increaseWeight(synapse);
                        else
                            decreaseWeight(synapse);
                    }
                }
            }

            // Lateral inhibition
            mOutputsIntegration.assign(
                {mOutputsDims[0], mOutputsDims[1], getNbOutputs(), 1}, 0.0);

            if (mInhibitRefractory > 0) {
                std::replace_if(mOutputsRefractoryEnd.begin(),
                                mOutputsRefractoryEnd.end(),
                                std::bind(std::less<Time_T>(),
                                          std::placeholders::_1,
                                          timestamp + mInhibitRefractory),
                                timestamp + mInhibitRefractory);
            }
        }

        refractoryEnd = timestamp + mRefractory; // Set the refractory period to
        // mRefractory, AFTER lateral
        // inhibition

        if (negSpike)
            integration += scaledThres;
        else
            integration -= scaledThres;

        mOutputs(subOx, subOy, output, 0)
            ->incomingSpike(NULL, timestamp + 1 * TimeFs, negSpike);
    }
}

void N2D2::ConvCell_Spike_RRAM::notify(Time_T timestamp, NotifyType notify)
{
    ConvCell_Spike::notify(timestamp, notify);

    if (notify == Reset)
        mInputsActivationTime.assign(mInputsActivationTime.dims(), 0);
}

N2D2::Synapse* N2D2::ConvCell_Spike_RRAM::newSynapse() const
{
    return new Synapse_RRAM(mBipolarWeights,
                            mSynapticRedundancy,
                            mIncomingDelay.spreadNormal(0),
                            mWeightsMinMean,
                            mWeightsMaxMean,
                            mWeightsMinVarSlope,
                            mWeightsMinVarOrigin,
                            mWeightsMaxVarSlope,
                            mWeightsMaxVarOrigin,
                            mWeightsSetProba,
                            mWeightsResetProba,
                            mWeightsRelInit.spreadNormal());
}

void N2D2::ConvCell_Spike_RRAM::increaseWeight(Synapse_RRAM* synapse) const
{
    for (unsigned int dev = 0; dev < mSynapticRedundancy; ++dev) {
        Synapse_RRAM::Device& device = synapse->devices[dev];

        // Extrinsic switching probability
        if (Random::randUniform() <= mLtpProba) {
            // Intrinsic switching probability
            synapse->setPulse(device);
            ++synapse->stats[dev].statsSetEvents;
        }
    }

    for (unsigned int dev = mSynapticRedundancy,
                      devSize = synapse->devices.size();
         dev < devSize;
         ++dev) {
        Synapse_RRAM::Device& device = synapse->devices[dev];

        // Extrinsic switching probability
        if (Random::randUniform() <= mLtpProba) {
            // Intrinsic switching probability
            synapse->resetPulse(device);
            ++synapse->stats[dev].statsResetEvents;
        }
    }
}

void N2D2::ConvCell_Spike_RRAM::decreaseWeight(Synapse_RRAM* synapse) const
{
    for (unsigned int dev = 0; dev < mSynapticRedundancy; ++dev) {
        Synapse_RRAM::Device& device = synapse->devices[dev];

        // Extrinsic switching probability
        if (Random::randUniform() <= mLtdProba) {
            // Intrinsic switching probability
            synapse->resetPulse(device);
            ++synapse->stats[dev].statsResetEvents;
        }
    }

    for (unsigned int dev = mSynapticRedundancy,
                      devSize = synapse->devices.size();
         dev < devSize;
         ++dev) {
        Synapse_RRAM::Device& device = synapse->devices[dev];

        // Extrinsic switching probability
        if (Random::randUniform() <= mLtdProba) {
            // Intrinsic switching probability
            synapse->setPulse(device);
            ++synapse->stats[dev].statsSetEvents;
        }
    }
}
