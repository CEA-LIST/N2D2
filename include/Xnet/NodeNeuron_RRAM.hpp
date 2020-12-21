/*
    (C) Copyright 2012 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_NODENEURON_RRAM_H
#define N2D2_NODENEURON_RRAM_H

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "NodeNeuron.hpp"
#include "Synapse_RRAM.hpp"
#include "utils/Gnuplot.hpp"

namespace N2D2 {
/**
 * Special LIF neuron class for stochastic learning with RRAM using STDP and
 * lateral inhibition (used in IEDM 2012).
*/
class NodeNeuron_RRAM : public NodeNeuron {
public:
    enum SynapseType {
        CBRAM,
        OXRAM
    };

    NodeNeuron_RRAM(Network& net);
    void propagateSpike(Node* origin, Time_T timestamp, EventType_T type = 0);
    void incomingSpike(Node* link, Time_T timestamp, EventType_T type = 0);
    void emitSpike(Time_T timestamp, EventType_T type = 0);
    void lateralInhibition(Time_T timestamp, EventType_T /*type*/ = 0);
    void reset(Time_T timestamp = 0);
    void logWeights(const std::string& fileName) const;
    void logSynapticBehavior(const std::string& fileName,
                             unsigned int nbSynapses = 100,
                             bool plot = true);
    void setSynapseType(SynapseType type)
    {
        mSynapseType = type;
    };
    virtual ~NodeNeuron_RRAM() {};

private:
    enum EventType {
        FireEvent = 1
    };

    void initialize();
    virtual Synapse* newSynapse() const;
    virtual void saveInternal(std::ofstream& dataFile) const;
    virtual void loadInternal(std::ifstream& dataFile);
    virtual void logStatePlot();
    void increaseWeight(Synapse_RRAM* synapse) const;
    void decreaseWeight(Synapse_RRAM* synapse) const;

    // Parameters
    /// Synaptic incoming delay \f$w_{delay}\f$
    ParameterWithSpread<Time_T> mIncomingDelay;
    /// Mean minimum synaptic weight for this neuron \f$\overline{w_{min}}\f$
    ParameterWithSpread<Weight_T> mWeightsMinMean;
    /// Mean maximum synaptic weight for this neuron \f$\overline{w_{max}}\f$
    ParameterWithSpread<Weight_T> mWeightsMaxMean;
    /// Mean initial synaptic weight for this neuron \f$\overline{w_{init}}\f$
    Parameter<Weight_T> mWeightsInitMean;
    /**
     * CBRAM specific parameter.
     * Intrinsic minimum conductance log-normal variability \f$w_{min_{var}}\f$
     * (expressed in standard deviation
     * of the log-normal distribution).
     * The device to device variability is log-normally distributed by
     * \f$ln\mathcal{N}\left(ln(\overline{w_{min}}),
     * {\sigma_{w_{min}}}^2\right)\f$ where \f$\overline{w_{min}}\f$ and
     * \f$\sigma_{w_{min}}\f$ are neural parameters. @n
     * With the addition of intrinsic variability, \f$w_{min}\f$ becomes
     * \f$w_{min_{mean}}\f$ and after each programming pulse,
     * \f$w_{min}\f$ changes according to the following distribution: @n
     * \f$w_{min} = ln\mathcal{N}\left(ln(w_{min_{mean}}),
     * {w_{min_{var}}}^2\right)\f$
     * with \f$w_{min_{var}} = \mathcal{N}(\overline{w_{min_{var}}},
     * {\sigma_{w_{min_{var}}}}^2)\f$ @n @n
     * Parameters \f$\overline{w_{min}}\f$, \f$\sigma_{w_{min}}\f$,
     * \f$\overline{w_{min_{var}}}\f$ and
     * \f$\sigma_{w_{min_{var}}}\f$ can be extracted from measurements with the
     * tools/spread.py script.
    */
    ParameterWithSpread<double> mWeightsMinVar;
    /**
     * CBRAM specific parameter.
     * Intrinsic maximum conductance log-normal variability \f$w_{max_{var}}\f$
     * (expressed in standard deviation
     * of the log-normal distribution).
     * The device to device variability is log-normally distributed by
     * \f$ln\mathcal{N}\left(ln(\overline{w_{max}}),
     * {\sigma_{w_{max}}}^2\right)\f$ where \f$\overline{w_{max}}\f$ and
     * \f$\sigma_{w_{max}}\f$ are neural parameters. @n
     * With the addition of intrinsic variability, \f$w_{max}\f$ becomes
     * \f$w_{max_{mean}}\f$ and after each programming pulse,
     * \f$w_{max}\f$ changes according to the following distribution: @n
     * \f$w_{max} = ln\mathcal{N}\left(ln(w_{max_{mean}}),
     * {w_{max_{var}}}^2\right)\f$
     * with \f$w_{max_{var}} = \mathcal{N}(\overline{w_{max_{var}}},
     * {\sigma_{w_{max_{var}}}}^2)\f$ @n @n
     * Parameters \f$\overline{w_{max}}\f$, \f$\sigma_{w_{max}}\f$,
     * \f$\overline{w_{max_{var}}}\f$ and
     * \f$\sigma_{w_{max_{var}}}\f$ can be extracted from measurements with the
     * tools/spread.py script.
    */
    ParameterWithSpread<double> mWeightsMaxVar;
    /**
     * OXRAM specific parameter.
    */
    Parameter<double> mWeightsMinVarSlope;
    /**
     * OXRAM specific parameter.
    */
    Parameter<double> mWeightsMinVarOrigin;
    /**
     * OXRAM specific parameter.
    */
    Parameter<double> mWeightsMaxVarSlope;
    /**
     * OXRAM specific parameter.
    */
    Parameter<double> mWeightsMaxVarOrigin;
    /**
     * Intrinsic SET switching probability \f$P_{SET}\f$ (upon receiving a SET
     * programming pulse).
     * Assuming uniform statistical distribution (not well supported by
     * experiments on RRAM).
    */
    ParameterWithSpread<double> mWeightsSetProba;
    /**
     * Intrinsic RESET switching probability \f$P_{RESET}\f$ (upon receiving a
     * RESET programming pulse).
     * Assuming uniform statistical distribution (not well supported by
     * experiments on RRAM).
    */
    ParameterWithSpread<double> mWeightsResetProba;
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
    /// Reduction of the neural integration through lateral inhibition
    /// (cumulative with other lateral inhibition processes)
    Parameter<double> mInhibitIntegration;

    /// Synaptic redundancy (number of RRAM device per synapse)
    Parameter<unsigned int> mSynapticRedundancy;
    /// Extrinsic STDP LTP probability (cumulative with intrinsic SET switching
    /// probability \f$P_{SET}\f$)
    Parameter<double> mLtpProba;
    /// Extrinsic STDP LTD probability (cumulative with intrinsic RESET
    /// switching probability \f$P_{RESET}\f$)
    Parameter<double> mLtdProba;

    // Internal variables
    SynapseType mSynapseType;
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
};
}

#endif // N2D2_NODENEURON_RRAM_H
