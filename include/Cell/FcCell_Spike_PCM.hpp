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

#ifndef N2D2_FCCELL_SPIKE_PCM_H
#define N2D2_FCCELL_SPIKE_PCM_H

#include "FcCell_Spike.hpp"
#include "Synapse_PCM.hpp"

namespace N2D2 {
class FcCell_Spike_PCM : public FcCell_Spike {
public:
    FcCell_Spike_PCM(Network& net,
                     const std::string& name,
                     unsigned int nbOutputs);
    static std::shared_ptr<FcCell> create(Network& net,
                                          const std::string& name,
                                          unsigned int nbOutputs,
                                          const std::shared_ptr
                                          <Activation<Float_T> >& /*activation*/
                                          = std::shared_ptr
                                          <Activation<Float_T> >())
    {
        return std::make_shared<FcCell_Spike_PCM>(net, name, nbOutputs);
    }

    void propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    void incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual ~FcCell_Spike_PCM() {};

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
    static Registrar<FcCell> mRegistrar;
};
}

#endif // N2D2_FCCELL_SPIKE_PCM_H
