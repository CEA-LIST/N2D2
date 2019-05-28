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

#ifndef N2D2_CONVCELL_SPIKE_PCM_H
#define N2D2_CONVCELL_SPIKE_PCM_H

#include "ConvCell_Spike.hpp"
#include "DeepNet.hpp"
#include "Synapse_PCM.hpp"

namespace N2D2 {
class ConvCell_Spike_PCM : public ConvCell_Spike {
public:
    ConvCell_Spike_PCM(Network& net, const DeepNet& deepNet, 
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
        return std::make_shared<ConvCell_Spike_PCM>(net, deepNet,
                                                    name,
                                                    kernelDims,
                                                    nbOutputs,
                                                    subSampleDims,
                                                    strideDims,
                                                    paddingDims,
                                                    dilationDims);
    }

    void propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    void incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual ~ConvCell_Spike_PCM() {};

private:
    Synapse* newSynapse() const;

    // Parameters
    /// Mean minimum synaptic weight \f$w_{min}\f$
    ParameterWithSpread<Weight_T> mWeightsMinMean;
    /// Mean maximum synaptic weight \f$w_{max}\f$
    ParameterWithSpread<Weight_T> mWeightsMaxMean;
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

    /// Synaptic redundancy (number of PCM device per synapse)
    Parameter<unsigned int> mSynapticRedundancy;
    Parameter<bool> mBipolarWeights;
    Parameter<bool> mBipolarIntegration;

private:
    static Registrar<ConvCell> mRegistrar;
};
}

#endif // N2D2_CONVCELL_SPIKE_PCM_H
