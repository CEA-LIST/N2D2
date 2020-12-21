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

#ifndef N2D2_CONVCELL_SPIKE_RRAM_H
#define N2D2_CONVCELL_SPIKE_RRAM_H

#include "ConvCell_Spike.hpp"
#include "DeepNet.hpp"
#include "Xnet/Synapse_RRAM.hpp"

namespace N2D2 {
class ConvCell_Spike_RRAM : public ConvCell_Spike {
public:
    ConvCell_Spike_RRAM(Network& net, const DeepNet& deepNet, 
                        const std::string& name,
                        const std::vector<unsigned int>& kernelDims,
                        unsigned int nbOutputs,
                        const std::vector<unsigned int>& subSampleDims
                                = std::vector<unsigned int>(2, 1U),
                        const std::vector<unsigned int>& strideDims
                                = std::vector<unsigned int>(2, 1U),
                        const std::vector<int>& paddingDims
                                = std::vector<int>(2, 0),
                        const std::vector<unsigned int>& dilationDims
                                = std::vector<unsigned int>(2, 1U));
    static std::shared_ptr<ConvCell>
    create(Network& net, const DeepNet& deepNet, 
           const std::string& name,
           const std::vector<unsigned int>& kernelDims,
           unsigned int nbOutputs,
           const std::vector<unsigned int>& subSampleDims
                = std::vector<unsigned int>(2, 1U),
           const std::vector<unsigned int>& strideDims
                = std::vector<unsigned int>(2, 1U),
           const std::vector<int>& paddingDims
                = std::vector<int>(2, 0),
           const std::vector<unsigned int>& dilationDims
                = std::vector<unsigned int>(2, 1U),
           const std::shared_ptr<Activation>& /*activation*/
           = std::shared_ptr<Activation>())
    {
        return std::make_shared<ConvCell_Spike_RRAM>(net, deepNet, 
                                                     name,
                                                     kernelDims,
                                                     nbOutputs,
                                                     subSampleDims,
                                                     strideDims,
                                                     paddingDims,
                                                     dilationDims);
    }

    virtual void initialize();
    void propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    void incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    void notify(Time_T timestamp, NotifyType notify);
    virtual ~ConvCell_Spike_RRAM() {};

private:
    Synapse* newSynapse() const;
    void increaseWeight(Synapse_RRAM* synapse) const;
    void decreaseWeight(Synapse_RRAM* synapse) const;

    // Parameters
    /// Mean minimum synaptic weight \f$w_{min}\f$
    ParameterWithSpread<Weight_T> mWeightsMinMean;
    /// Mean maximum synaptic weight \f$w_{max}\f$
    ParameterWithSpread<Weight_T> mWeightsMaxMean;
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

    /// Synaptic redundancy (number of RRAM device per synapse)
    Parameter<unsigned int> mSynapticRedundancy;
    Parameter<bool> mBipolarWeights;
    Parameter<bool> mBipolarIntegration;
    /// Extrinsic STDP LTP probability (cumulative with intrinsic SET switching
    /// probability \f$P_{SET}\f$)
    Parameter<double> mLtpProba;
    /// Extrinsic STDP LTD probability (cumulative with intrinsic RESET
    /// switching probability \f$P_{RESET}\f$)
    Parameter<double> mLtdProba;
    /// STDP LTP time window \f$T_{LTP}\f$
    Parameter<Time_T> mStdpLtp;
    /// Neural lateral inhibition period \f$T_{inhibit}\f$
    Parameter<Time_T> mInhibitRefractory;
    /// If false, STDP is disabled (no synaptic weight change)
    Parameter<bool> mEnableStdp;
    Parameter<bool> mRefractoryIntegration;
    Parameter<bool> mDigitalIntegration;

    Tensor<Time_T> mInputsActivationTime;

private:
    static Registrar<ConvCell> mRegistrar;
};
}

#endif // N2D2_CONVCELL_SPIKE_RRAM_H
