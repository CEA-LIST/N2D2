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

#ifndef N2D2_NODENEURON_PCM_H
#define N2D2_NODENEURON_PCM_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "NodeNeuron.hpp"
#include "Synapse_PCM.hpp"
#include "utils/Gnuplot.hpp"

namespace N2D2 {
/**
 * Special LIF neuron class for learning with the 2-PCM synapse using STDP and
 * lateral inhibition (used in IEDM 2011).
*/
class NodeNeuron_PCM : public NodeNeuron {
public:
    enum DeviceId {
        LTP = 0,
        LTD = 1
    };
    enum WeightsRefreshMethod {
        Nearest,
        Truncate
    };

    NodeNeuron_PCM(Network& net);
    void propagateSpike(Node* origin, Time_T timestamp, EventType_T type = 0);
    void incomingSpike(Node* link, Time_T timestamp, EventType_T type = 0);
    void emitSpike(Time_T timestamp, EventType_T type = 0);
    void lateralInhibition(Time_T timestamp, EventType_T /*type*/ = 0);
    void reset(Time_T timestamp = 0);
    double getRelativeLtpStrength() const
    {
        return mRelativeLtpStrength;
    };
    void logSynapticBehavior(const std::string& fileName,
                             unsigned int nbSynapses = 100,
                             bool plot = true) const;

    /**
     * Load experimental potentiating measurements for PCM corresponding to
     *successives identical potentiating pulses.
     * If data is loaded, the synaptic model parameters \f$\alpha{}_{+}\f$ and
     *\f$\beta{}_{+}\f$, defined in NodeNeuron, are
     * ignored.
     * @warning The loaded data is always normalized by the synapse
     *\f$w_{min}\f$ and \f$w_{max}\f$ parameters. The absolute
     * values of the conductance in @p fileName are thus ignored. As a
     *consequence, synaptic variability, defined at the neuron
     * level, is preserved.
     *
     * @param   fileName        Data file containing a list of conductance value
     *for successives potentiating pulses
    */
    void experimentalLtpModel(const std::string& fileName);
    /// Returns the LTP experimental model used for the synaptic updates, if
    /// loaded with NodeNeuron_PCM::experimentalLtpModel()
    const std::vector<double>& getExperimentalLtpModel() const
    {
        return mExperimentalLtpModel;
    };
    /// Destructor.
    virtual ~NodeNeuron_PCM() {};

private:
    enum EventType {
        FireEvent = 1
    };

    void initialize();
    virtual Synapse* newSynapse() const;
    virtual void saveInternal(std::ofstream& dataFile) const;
    virtual void loadInternal(std::ifstream& dataFile);
    virtual void logStatePlot();
    void increaseWeight(Synapse_PCM* synapse, DeviceId dev) const;
    void decreaseWeight(Synapse_PCM* synapse, DeviceId dev) const;

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
    /**
     * Add variability at each PCM potentiation, or crystallization (in percent
     * of the nominal increment step).
     * \f$\Delta{}w = \mathcal{N}(\Delta{}w,
     * {(\Delta{}w.mWeightIncrementVar)}^2)\f$
    */
    Parameter<double> mWeightIncrementVar;
    /// Synaptic weight increment damping factor for this neuron
    /// \f$\beta{}_{+}\f$
    ParameterWithSpread<double> mWeightIncrementDamping;
    /// If true, PCM device parameters @p weightMin, @p weightMax, @p
    /// weightIncrement and @p weightIncrementDamping are
    /// randomized at each amorphization (= refresh)
    Parameter<bool> mWeightStochastic;
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
    /// If false, STDP is disabled (no synaptic weight change)
    Parameter<bool> mEnableStdp;
    /// Relative LTP strength. The contribution to the neural integration of LTP
    /// devices is @p mRelativeLtpStrength times the
    /// contribution of the LTD devices.
    Parameter<double> mRelativeLtpStrength;
    /// Number of synaptic updates (= number of neural activations), before the
    /// synapses are refreshed
    Parameter<unsigned int> mWeightsUpdateLimit;
    /// Weight refresh method, Truncate only resets one or the two devices and
    /// Nearest reprogram the equivalent synaptic weight
    Parameter<WeightsRefreshMethod> mWeightsRefreshMethod;

    // Internal variables
    /// Neuron's integration, or membrane potential
    double mIntegration;
    /// Indicates if the neuron is allowed to fire
    bool mAllowFire;
    /// Last incoming spike time
    Time_T mLastSpikeTime;
    /// Address to the last event emitted by the neuron (NULL = no event)
    SpikeEvent* mEvent;
    /// Programmed end of the refractory period of the neuron, either caused by
    /// its activation, or lateral inhibition
    Time_T mRefractoryEnd;
    /// Number of synaptic updates (= number of neural activations with STDP
    /// enabled), used to implement the refresh protocol
    unsigned int mWeightUpdate;
    /// If not empty, the values in the vector are used for the successive
    /// conductance changes through potentiating pulses
    std::vector<double> mExperimentalLtpModel;
};
}

namespace {
template <>
const char* const EnumStrings
    <N2D2::NodeNeuron_PCM::WeightsRefreshMethod>::data[]
    = {"Nearest", "Truncate"};
}

#endif // N2D2_NODENEURON_PCM_H
