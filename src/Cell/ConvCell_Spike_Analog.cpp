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

#include "Cell/ConvCell_Spike_Analog.hpp"

N2D2::Registrar<N2D2::ConvCell>
N2D2::ConvCell_Spike_Analog::mRegistrar("Spike_Analog",
                                        N2D2::ConvCell_Spike_Analog::create);

N2D2::ConvCell_Spike_Analog::ConvCell_Spike_Analog(Network& net,
                                                   const std::string& name,
                                                   unsigned int kernelWidth,
                                                   unsigned int kernelHeight,
                                                   unsigned int nbOutputs,
                                                   unsigned int subSampleX,
                                                   unsigned int subSampleY,
                                                   unsigned int strideX,
                                                   unsigned int strideY,
                                                   int paddingX,
                                                   int paddingY)
    : Cell(name, nbOutputs),
      ConvCell(name,
               kernelWidth,
               kernelHeight,
               nbOutputs,
               subSampleX,
               subSampleY,
               strideX,
               strideY,
               paddingX,
               paddingY),
      ConvCell_Spike(net,
                     name,
                     kernelWidth,
                     kernelHeight,
                     nbOutputs,
                     subSampleX,
                     subSampleY,
                     strideX,
                     strideY,
                     paddingX,
                     paddingY),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mWeightsMinMean(this, "WeightsMinMean", 1, 0.1),
      mWeightsMaxMean(this, "WeightsMaxMean", 100, 10.0),
      mWeightIncrement(this, "WeightIncrement", 0.5, 0.05),
      mWeightIncrementDamping(this, "WeightIncrementDamping", -3.0),
      mWeightDecrement(this, "WeightDecrement", 0.5, 0.05),
      mWeightDecrementDamping(this, "WeightDecrementDamping", -3.0),
      mBipolarIntegration(this, "BipolarIntegration", false),
      mStdpLtp(this, "StdpLtp", 1000 * TimePs),
      mInhibitRefractory(this, "InhibitRefractory", 0 * TimeS),
      mEnableStdp(this, "EnableStdp", true),
      mRefractoryIntegration(this, "RefractoryIntegration", true)
{
    // ctor
}

void N2D2::ConvCell_Spike_Analog::initialize()
{
    ConvCell_Spike::initialize();

    mInputsActivationTime.resize(
        mChannelsWidth, mChannelsHeight, mNbChannels, 1);
    mInputsActivity.resize(mChannelsWidth, mChannelsHeight, mNbChannels, 1);
}

void N2D2::ConvCell_Spike_Analog::propagateSpike(NodeIn* origin,
                                                 Time_T timestamp,
                                                 EventType_T type)
{
    const Area& area = origin->getArea();
    const unsigned int channel = origin->getChannel();

    mInputsActivationTime(area.x, area.y, channel, 0) = timestamp;

    if (type)
        --mInputsActivity(area.x, area.y, channel, 0);
    else
        ++mInputsActivity(area.x, area.y, channel, 0);

    const unsigned int oxStride
        = mStrideX
          * (unsigned int)((mChannelsWidth + 2 * mPaddingX - mKernelWidth
                            + mStrideX) / (double)mStrideX);
    const unsigned int oyStride
        = mStrideY
          * (unsigned int)((mChannelsHeight + 2 * mPaddingY - mKernelHeight
                            + mStrideY) / (double)mStrideY);
    const unsigned int ixPad = area.x + mPaddingX;
    const unsigned int iyPad = area.y + mPaddingY;
    const unsigned int sxMax = std::min(mKernelWidth, ixPad + 1);
    const unsigned int syMax = std::min(mKernelHeight, iyPad + 1);

    for (unsigned int sy = iyPad % mStrideY, sx0 = ixPad % mStrideX; sy < syMax;
         sy += mStrideY) {
        if (iyPad >= oyStride + sy)
            continue;

        for (unsigned int sx = sx0; sx < sxMax; sx += mStrideX) {
            // Border conditions
            if (ixPad >= oxStride + sx)
                continue;

            // Output node coordinates
            const unsigned int ox = (ixPad - sx) / mStrideX;
            const unsigned int oy = (iyPad - sy) / mStrideY;

            for (unsigned int output = 0; output < mNbOutputs; ++output) {
                if (!isConnection(channel, output))
                    continue;

                const Time_T delay = static_cast<Synapse_Behavioral*>(
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

void N2D2::ConvCell_Spike_Analog::incomingSpike(NodeIn* origin,
                                                Time_T timestamp,
                                                EventType_T type)
{
    // Output node coordinates
    unsigned int output, ox, oy;
    bool negative;
    std::tie(output, ox, oy, negative) = unmaps(type);

    const unsigned int subOx = ox / mSubSampleX;
    const unsigned int subOy = oy / mSubSampleY;

    // Synapse coordinates
    const Area& area = origin->getArea();
    const unsigned int synX = area.x - ox * mStrideX + mPaddingX;
    const unsigned int synY = area.y - oy * mStrideY + mPaddingY;

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

    Synapse_Behavioral* synapse = static_cast<Synapse_Behavioral*>(
        mSharedSynapses(synX, synY, origin->getChannel(), output));

    if (mBipolarIntegration) {
        // For off-line learning, spike-based feed-forward, without bipolar
        // synapses
        integration
            += (negative) ? -2.0 * synapse->weight
                            + (mWeightsMaxMean.mean() + mWeightsMinMean.mean())
                          : 2.0 * synapse->weight
                            - (mWeightsMaxMean.mean() + mWeightsMinMean.mean());
    } else
        integration += (negative) ? -synapse->weight : synapse->weight;

    // Stats
    ++synapse->statsReadEvents;

    // For STDP, integration stays at 0 during refractory period
    // For off-line learning, integration must continue during refractory period
    if (!mRefractoryIntegration && timestamp < refractoryEnd)
        integration = 0.0;

    if ((integration >= mThreshold
         || (mBipolarThreshold && (-integration) >= mThreshold))
        && timestamp >= refractoryEnd) {
        const bool negSpike = (integration < 0);

        if (mEnableStdp) {
            const unsigned int sxMin = (unsigned int)std::max(
                (int)mPaddingX - (int)(ox * mStrideX), 0);
            const unsigned int syMin = (unsigned int)std::max(
                (int)mPaddingY - (int)(oy * mStrideY), 0);
            const unsigned int sxMax = Utils::clamp<int>(
                mChannelsWidth + mPaddingX - ox * mStrideX, 0, mKernelWidth);
            const unsigned int syMax = Utils::clamp<int>(
                mChannelsHeight + mPaddingY - oy * mStrideY, 0, mKernelHeight);

            for (unsigned int channel = 0; channel < mNbChannels; ++channel) {
                if (!isConnection(channel, output))
                    continue;

                for (unsigned int sy = syMin; sy < syMax; ++sy) {
                    for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                        const unsigned int ix = (int)(ox * mStrideX + sx)
                                                - (int)mPaddingX;
                        const unsigned int iy = (int)(oy * mStrideY + sy)
                                                - (int)mPaddingY;

                        const Time_T lastSpike
                            = mInputsActivationTime(ix, iy, channel, 0);
                        Synapse_Behavioral* synapse = static_cast
                            <Synapse_Behavioral*>(
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
                mOutputsWidth, mOutputsHeight, mNbOutputs, 1, 0.0);

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
            integration += mThreshold;
        else
            integration -= mThreshold;

        mOutputs(subOx, subOy, output, 0)
            ->incomingSpike(NULL, timestamp + 1 * TimeFs, negSpike);
    }
}

void N2D2::ConvCell_Spike_Analog::notify(Time_T timestamp, NotifyType notify)
{
    ConvCell_Spike::notify(timestamp, notify);

    if (notify == Reset) {
        mInputsActivationTime.assign(
            mChannelsWidth, mChannelsHeight, mNbChannels, 1, 0);
        mInputsActivity.assign(
            mChannelsWidth, mChannelsHeight, mNbChannels, 1, 0);
    }
}

N2D2::Synapse* N2D2::ConvCell_Spike_Analog::newSynapse() const
{
    return new Synapse_Behavioral(mIncomingDelay.spreadNormal(0),
                                  mWeightsMinMean.spreadNormal(0),
                                  mWeightsMaxMean.spreadNormal(0),
                                  mWeightIncrement.spreadNormal(),
                                  mWeightIncrementDamping.spreadNormal(),
                                  mWeightDecrement.spreadNormal(),
                                  mWeightDecrementDamping.spreadNormal(),
                                  mWeightsRelInit.spreadNormal(0));
}

void N2D2::ConvCell_Spike_Analog::increaseWeight(Synapse_Behavioral
                                                 * synapse) const
{
    if (synapse->weight < synapse->weightMax) {
        const double dw
            = synapse->weightIncrement
              * std::exp(synapse->weightIncrementDamping
                         * (synapse->weight - synapse->weightMin)
                         / (synapse->weightMax - synapse->weightMin));

        if (synapse->weight < synapse->weightMax - dw)
            synapse->weight += dw;
        else
            synapse->weight = synapse->weightMax;
    }

    ++synapse->statsIncEvents;
}

void N2D2::ConvCell_Spike_Analog::decreaseWeight(Synapse_Behavioral
                                                 * synapse) const
{
    if (synapse->weight > synapse->weightMin) {
        const double dw
            = synapse->weightDecrement
              * std::exp(synapse->weightDecrementDamping
                         * (synapse->weightMax - synapse->weight)
                         / (synapse->weightMax - synapse->weightMin));

        if (synapse->weight > synapse->weightMin + dw)
            synapse->weight -= dw;
        else
            synapse->weight = synapse->weightMin;
    }

    ++synapse->statsDecEvents;
}
