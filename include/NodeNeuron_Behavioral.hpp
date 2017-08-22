/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

#ifndef N2D2_NODENEURON_BEHAVIORAL_H
#define N2D2_NODENEURON_BEHAVIORAL_H

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>

#include "NodeNeuron.hpp"
#include "Synapse_Behavioral.hpp"

namespace N2D2 {
/**
 * As the name suggests it, it's a neuron (IF or LIF). It is highly customizable
 * and implements STDP and lateral inhibition.
*/
class NodeNeuron_Behavioral : public NodeNeuron {
public:
    NodeNeuron_Behavioral(Network& net);
    void propagateSpike(Node* origin, Time_T timestamp, EventType_T type = 0);
    void incomingSpike(Node* link, Time_T timestamp, EventType_T type = 0);
    void emitSpike(Time_T timestamp, EventType_T type = 0);
    void lateralInhibition(Time_T timestamp, EventType_T /*type*/ = 0);
    void reset(Time_T timestamp = 0);
    Time_T getRefractoryEnd() const
    {
        return mRefractoryEnd;
    };
    void setRefractoryEnd(Time_T refractoryEnd)
    {
        mRefractoryEnd = refractoryEnd;
    };
    void logSynapticBehavior(const std::string& fileName,
                             unsigned int nbSynapses = 100,
                             bool plot = true) const;
    void logStdpBehavior(const std::string& fileName,
                         unsigned int nbPoints = 100,
                         bool plot = true);
    virtual ~NodeNeuron_Behavioral() {};

    /**
     * Returns current integration (or membrane potential) of the neuron.
     * @return Current integration value
    */
    double getIntegration() const
    {
        return mIntegration;
    };

    /**
     * Sets the integration (or membrane potential) value of the neuron.
     * @param integration       Integration value
    */
    void setIntegration(double integration)
    {
        mIntegration = integration;
    };

    /**
     * Returns last incoming spike time of the neuron.
     * @return Last incoming spike time value
    */
    Time_T getLastSpikeTime() const
    {
        return mLastSpikeTime;
    };

    /**
     * Sets the last incoming spike time of the neuron.
     * @param lastSpikeTime     Last incoming spike time value
    */
    void setLastSpikeTime(Time_T lastSpikeTime)
    {
        mLastSpikeTime = lastSpikeTime;
    };

private:
    enum EventType {
        FireEvent = 1
    };

    void initialize();
    virtual Synapse* newSynapse() const;
    virtual void saveInternal(std::ofstream& dataFile) const;
    virtual void loadInternal(std::ifstream& dataFile);
    virtual void logStatePlot();
    bool
    stdp(Synapse_Behavioral* synapse, Time_T preTime, Time_T postTime) const;
    void increaseWeight(Synapse_Behavioral* synapse,
                        double weightIncrement) const;
    void decreaseWeight(Synapse_Behavioral* synapse,
                        double weightDecrement) const;

    // Parameters
    /// Synaptic incoming delay \f$w_{delay}\f$
    ParameterWithSpread<Time_T> mIncomingDelay;
    /// Minimum synaptic weight for this neuron \f$w_{min}\f$
    ParameterWithSpread<Weight_T> mWeightsMin;
    /// Maximum synaptic weight for this neuron \f$w_{max}\f$
    ParameterWithSpread<Weight_T> mWeightsMax;
    /// Initial synaptic weight for this neuron \f$w_{init}\f$
    ParameterWithSpread<Weight_T> mWeightsInit;
    /// Synaptic weight increment for this neuron \f$\alpha{}_{+}\f$
    ParameterWithSpread<Weight_T> mWeightIncrement;
    /// Synaptic weight increment damping factor for this neuron
    /// \f$\beta{}_{+}\f$
    ParameterWithSpread<double> mWeightIncrementDamping;
    /// Synaptic weight decrement for this neuron \f$\alpha{}_{-}\f$
    ParameterWithSpread<Weight_T> mWeightDecrement;
    /// Synaptic weight decrement damping factor for this neuron
    /// \f$\beta{}_{-}\f$
    ParameterWithSpread<double> mWeightDecrementDamping;
    /// Threshold of the neuron \f$I_{thres}\f$
    ParameterWithSpread<double> mThreshold;
    /// Emit delay of the neuron \f$T_{emit}\f$
    ParameterWithSpread<Time_T> mEmitDelay;
    /// If true, a call to reset() is necessary to re-enable the neuron once it
    /// fired
    Parameter<bool> mFireOnlyOnce;
    /// STDP LTP time window \f$T_{LTP}\f$
    ParameterWithSpread<Time_T> mStdpLtp;
    /// Neural leak time constant \f$\tau_{leak}\f$ (if 0, no leak)
    ParameterWithSpread<Time_T> mLeak;
    /// Neural refractory period \f$T_{refrac}\f$
    ParameterWithSpread<Time_T> mRefractory;
    /// Neural lateral inhibition period \f$T_{inhibit}\f$
    ParameterWithSpread<Time_T> mInhibitRefractory;
    Parameter<double> mInhibitIntegration;
    Parameter<unsigned int> mInhibitStdp;
    /// If false, STDP is disabled (no synaptic weight change)
    Parameter<bool> mEnableStdp;
    Parameter<unsigned int> mOrderStdp;
    Parameter<bool> mLinearLeak;
    Parameter<Weight_T> mWeightBias;
    Parameter<Time_T> mStdpLtd;
    Parameter<bool> mBiologicalStdp;

    // Internal variables
    /// Neuron's integration, or membrane potential
    double mIntegration;
    /// Indicates if the neuron is allowed to fire
    bool mAllowFire;
    bool mAllowStdp;
    /// Number of inhibitions through lateral connections, used for @p
    /// mInhibitStdp and @p mInhibitFire
    unsigned int mInhibition;
    /// Last incoming spike time
    Time_T mLastSpikeTime;
    /// Address to the last event emitted by the neuron (NULL = no event)
    SpikeEvent* mEvent;
    /// Programmed end of the refractory period of the neuron, either caused by
    /// its activation, or lateral inhibition
    Time_T mRefractoryEnd;
    Time_T mLastStdp;
    std::deque<Synapse_Behavioral*> mLtpFifo;
};
}

#endif // N2D2_NODENEURON_BEHAVIORAL_H
