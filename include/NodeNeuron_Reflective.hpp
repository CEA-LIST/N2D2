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

#ifndef N2D2_NODENEURON_REFLECTIVE_H
#define N2D2_NODENEURON_REFLECTIVE_H

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>

#include "NodeNeuron.hpp"
#include "Synapse_Behavioral.hpp"

namespace N2D2 {
/**
 * Special LIF neuron class for the construction of bi-directionnal spiking
 * neural network, with special features like pattern
 * recall.
*/
class NodeNeuron_Reflective : public NodeNeuron {
public:
    enum ReflectiveType {
        None = 0,
        Forward,
        Backward,
        Both
    };

    NodeNeuron_Reflective(Network& net);
    virtual void
    propagateSpike(Node* origin, Time_T timestamp, EventType_T type);
    virtual void
    incomingSpike(Node* link, Time_T timestamp, EventType_T type = 0);
    void emitSpike(Time_T timestamp, EventType_T type = 0);
    void lateralInhibition(Time_T timestamp, EventType_T type);
    virtual void initialize();
    void reset(Time_T timestamp = 0);
    virtual ~NodeNeuron_Reflective() {};

protected:
    enum EventType {
        ForwardEvent = 1,
        BackwardEvent,
        ForwardSubEvent,
        BackwardSubEvent,
        ForwardEchoEvent,
        BackwardEchoEvent
    };

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
    ParameterWithSpread<double> mBackwardThreshold;
    Parameter<Time_T> mBackwardStdpLtp;
    Parameter<Time_T> mBackwardLeak;
    Parameter<Time_T> mBackwardRefractory;
    Parameter<Time_T> mBackwardInhibitRefractory;
    Parameter<double> mBackwardSubActivity;
    Parameter<Time_T> mBackwardEchoMeanPeriod;
    Parameter<Time_T> mBackwardEchoMeanDuration;
    Parameter<bool> mBackwardSubEventStdp;
    Parameter<bool> mBackwardEchoEventStdp;

    Parameter<ReflectiveType> mBackwardPropagation;
    Parameter<ReflectiveType> mStdpLearning;

private:
    Synapse* newSynapse() const;
    virtual void saveInternal(std::ofstream& dataFile) const;
    virtual void loadInternal(std::ifstream& dataFile);
    virtual void logStatePlot();
    void increaseWeight(Synapse_Behavioral* synapse) const;
    void decreaseWeight(Synapse_Behavioral* synapse) const;

    // Parameters
    Parameter<Time_T> mStdpLtp;
    Parameter<Time_T> mLeak;
    Parameter<Time_T> mRefractory;
    Parameter<Time_T> mInhibitRefractory;
    Parameter<double> mSubActivity;
    Parameter<Time_T> mEchoMeanPeriod;
    Parameter<Time_T> mEchoMeanDuration;
    Parameter<bool> mSubEventStdp;
    Parameter<bool> mEchoEventStdp;

    Parameter<ReflectiveType> mForwardPropagation;
    Parameter<bool> mShareIntegration;

    // Internal variables
    /// Neuron's integration, or membrane potential
    double mIntegration;
    /// Last incoming spike time
    Time_T mLastSpikeTime;
    /// Address to the last event emitted by the neuron (NULL = no event)
    SpikeEvent* mEvent;
    Time_T mRefractoryEnd;
    Time_T mEchoEnd;

    Time_T mBackwardRefractoryEnd;
    double mBackwardIntegration;
    Time_T mBackwardLastSpikeTime;
    SpikeEvent* mBackwardEvent;
    Time_T mBackwardEchoEnd;
};

class NodeNeuron_ReflectiveBridge : public NodeNeuron_Reflective {
public:
    NodeNeuron_ReflectiveBridge(Network& net);
    void addLink(Node* origin);
    void propagateSpike(Node* origin, Time_T timestamp, EventType_T type);
    void incomingSpike(Node* link, Time_T timestamp, EventType_T type = 0);
    virtual ~NodeNeuron_ReflectiveBridge() {};

private:
    void initialize();

    Node* mLink;
};
}

namespace {
template <>
const char* const EnumStrings
    <N2D2::NodeNeuron_Reflective::ReflectiveType>::data[]
    = {"None", "Forward", "Backward", "Both"};
}

#endif // N2D2_NODENEURON_REFLECTIVE_H
