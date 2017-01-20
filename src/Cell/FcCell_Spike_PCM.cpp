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

#include "Cell/FcCell_Spike_PCM.hpp"

N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Spike_PCM::mRegistrar("Spike_PCM", N2D2::FcCell_Spike_PCM::create);

N2D2::FcCell_Spike_PCM::FcCell_Spike_PCM(Network& net,
                                         const std::string& name,
                                         unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      FcCell(name, nbOutputs),
      FcCell_Spike(net, name, nbOutputs),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mWeightsMinMean(this, "WeightsMinMean", 1, 0.1),
      mWeightsMaxMean(this, "WeightsMaxMean", 100, 10.0),
      mWeightIncrement(this, "WeightIncrement", 50, 5.0),
      mWeightIncrementVar(this, "WeightIncrementVar", 0.0),
      mWeightIncrementDamping(this, "WeightIncrementDamping", -3.0),
      mSynapticRedundancy(this, "SynapticRedundancy", 1U),
      mBipolarWeights(this, "BipolarWeights", false),
      mBipolarIntegration(this, "BipolarIntegration", false)
{
    // ctor
}

void N2D2::FcCell_Spike_PCM::propagateSpike(NodeIn* origin,
                                            Time_T timestamp,
                                            EventType_T type)
{
    const Area& area = origin->getArea();

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        const Time_T delay = static_cast<Synapse_PCM*>(
            mSynapses(area.x, area.y, origin->getChannel(), output))->delay;

        if (delay > 0)
            mNet.newEvent(origin, NULL, timestamp + delay, maps(output, type));
        else
            incomingSpike(origin, timestamp, maps(output, type));
    }
}

void N2D2::FcCell_Spike_PCM::incomingSpike(NodeIn* origin,
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

    Synapse_PCM* synapse = static_cast
        <Synapse_PCM*>(mSynapses(area.x, area.y, origin->getChannel(), output));

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

    // Stats
    ++synapse->statsReadEvents;

    for (unsigned int dev = 0, devSize = synapse->devices.size(); dev < devSize;
         ++dev)
        synapse->stats[dev].statsReadEnergy += synapse->devices[dev].weight;

    const double scaledThres = mThreshold * mSynapticRedundancy;

    if ((integration >= scaledThres
         || (mBipolarThreshold && (-integration) >= scaledThres))
        && timestamp >= refractoryEnd) {
        const bool negSpike = (integration < 0);

        refractoryEnd = timestamp + mRefractory; // Set the refractory period to
        // mRefractory, AFTER lateral
        // inhibition

        if (negSpike)
            integration += scaledThres;
        else
            integration -= scaledThres;

        mOutputs(output)->incomingSpike(NULL, timestamp + 1 * TimeFs, negSpike);

        if (negSpike)
            --mNbActivations[output];
        else
            ++mNbActivations[output];

        if (mTerminateDelta > 0) {
            std::vector<int> nbActivations(mNbActivations);
            std::partial_sort(nbActivations.begin(),
                              nbActivations.begin() + 2,
                              nbActivations.end(),
                              std::greater<int>());

            if (nbActivations[0] - nbActivations[1] >= (int)mTerminateDelta)
                mNet.stop(timestamp + 2 * TimeFs, true);
        }
    }
}

N2D2::Synapse* N2D2::FcCell_Spike_PCM::newSynapse() const
{
    return new Synapse_PCM(mBipolarWeights,
                           mSynapticRedundancy,
                           mIncomingDelay.spreadNormal(0),
                           mWeightsMinMean,
                           mWeightsMaxMean,
                           mWeightIncrement,
                           mWeightIncrementVar,
                           mWeightIncrementDamping,
                           NULL,
                           mWeightsRelInit.spreadNormal());
}
