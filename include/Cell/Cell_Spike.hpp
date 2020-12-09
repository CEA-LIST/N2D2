/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CELL_SPIKE_H
#define N2D2_CELL_SPIKE_H

#include "Cell.hpp"
#include "Xnet/Network.hpp"
#include "Xnet/Synapse.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {

class DeepNet;
class NodeIn;
class NodeOut;

class Cell_Spike : public virtual Cell, public NetworkObserver {
public:
    Cell_Spike(Network& net, const DeepNet& deepNet, const std::string& name, 
               unsigned int nbOutputs);
    
    virtual void addInput(StimuliProvider& sp,
                          unsigned int channel,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width,
                          unsigned int height,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(StimuliProvider& sp,
                          unsigned int x0 = 0,
                          unsigned int y0 = 0,
                          unsigned int width = 0,
                          unsigned int height = 0,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(Cell* cell,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(Cell* cell,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width = 0,
                          unsigned int height = 0);
    
    virtual void clearInputs();

    inline virtual void
    propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void
    incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0) = 0;
    NodeOut* getOutput(unsigned int output) const
    {
        return mOutputs.at(output);
    }
    const std::vector<NodeOut*>& getOutputs() const
    {
        return mOutputs.data();
    };
    virtual Synapse::Stats logStats(const std::string& /*dirName*/) const
    {
        return Synapse::Stats();
    };
    virtual void spikeCodingCompare(const std::string& /*fileName*/) const {};
    virtual ~Cell_Spike();

protected:
    void populateOutputs();

    /// Synaptic incoming delay \f$w_{delay}\f$
    ParameterWithSpread<Time_T> mIncomingDelay;

    // Internal
    Network& mNet;
    // Forward
    std::vector<NodeIn*> mInputs;
    Tensor<NodeOut*> mOutputs;
};
}

void N2D2::Cell_Spike::propagateSpike(NodeIn* origin,
                                      Time_T timestamp,
                                      EventType_T type)
{
    incomingSpike(origin, timestamp, type);
}

#endif // N2D2_CELL_SPIKE_H
