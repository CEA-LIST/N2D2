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

#ifndef N2D2_FCCELL_SPIKE_H
#define N2D2_FCCELL_SPIKE_H

#include "Xcell.hpp"

#include "Cell_Spike.hpp"
#include "FcCell.hpp"
#include "NodeIn.hpp"
#include "NodeOut.hpp"

namespace N2D2 {
class FcCell_Spike : public virtual FcCell, public Cell_Spike {
public:
    FcCell_Spike(Network& net, const std::string& name, unsigned int nbOutputs);
    static std::shared_ptr<FcCell> create(Network& net,
                                          const std::string& name,
                                          unsigned int nbOutputs,
                                          const std::shared_ptr
                                          <Activation<Float_T> >& /*activation*/
                                          = std::shared_ptr
                                          <Activation<Float_T> >())
    {
        return std::make_shared<FcCell_Spike>(net, name, nbOutputs);
    }

    virtual void initialize();
    virtual void
    propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void
    incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void notify(Time_T timestamp, NotifyType notify);
    inline Float_T getWeight(unsigned int output, unsigned int channel) const;
    inline Float_T getBias(unsigned int /*output*/) const
    {
        return 0.0;
    };
    NodeId_T getBestResponseId(bool report = false) const;
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    Synapse::Stats logStats(const std::string& dirName) const;
    virtual ~FcCell_Spike();

protected:
    virtual Synapse* newSynapse() const;
    inline void
    setWeight(unsigned int output, unsigned int channel, Float_T value);
    inline void setBias(unsigned int /*output*/, Float_T /*value*/) {};
    EventType_T maps(unsigned int output, bool negative) const
    {
        return ((EventType_T)output << 1) | (int)negative;
    };
    std::pair<unsigned int, bool> unmaps(EventType_T type) const
    {
        return std::make_pair(type >> 1, type & 1);
    };

    /// Relative initial synaptic weight \f$w_{init}\f$
    ParameterWithSpread<Weight_T> mWeightsRelInit;

    /// Threshold of the neuron \f$I_{thres}\f$
    Parameter<double> mThreshold;
    Parameter<bool> mBipolarThreshold;
    /// Neural leak time constant \f$\tau_{leak}\f$ (if 0, no leak)
    Parameter<Time_T> mLeak;
    /// Neural refractory period \f$T_{refrac}\f$
    Parameter<Time_T> mRefractory;
    Parameter<unsigned int> mTerminateDelta;
    Parameter<unsigned int> mTerminateMax;

    // mSynapses[output node][input node]
    Tensor4d<Synapse*> mSynapses;

    std::vector<Time_T> mOutputsLastIntegration;
    std::vector<double> mOutputsIntegration;
    std::vector<Time_T> mOutputsRefractoryEnd;
    std::vector<int> mNbActivations;

private:
    static Registrar<FcCell> mRegistrar;
};
}

void N2D2::FcCell_Spike::setWeight(unsigned int output,
                                   unsigned int channel,
                                   Float_T value)
{
    mSynapses(channel, output)->setRelativeWeight(value);
}

N2D2::Float_T N2D2::FcCell_Spike::getWeight(unsigned int output,
                                            unsigned int channel) const
{
    return mSynapses(channel, output)->getRelativeWeight(true);
}

#endif // N2D2_FCCELL_SPIKE_H
