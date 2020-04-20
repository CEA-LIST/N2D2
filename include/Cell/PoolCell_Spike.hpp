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

#ifndef N2D2_POOLCELL_SPIKE_H
#define N2D2_POOLCELL_SPIKE_H

#include "Cell_Spike.hpp"
#include "DeepNet.hpp"
#include "PoolCell.hpp"

namespace N2D2 {
class PoolCell_Spike : public virtual PoolCell, public Cell_Spike {
public:
    PoolCell_Spike(Network& net,
                   const DeepNet& deepNet, 
                   const std::string& name,
                   const std::vector<unsigned int>& poolDims,
                   unsigned int nbOutputs,
                   const std::vector<unsigned int>& strideDims
                      = std::vector<unsigned int>(2, 1U),
                   const std::vector<unsigned int>& paddingDims
                      = std::vector<unsigned int>(2, 0),
                   Pooling pooling = Max);
    static std::shared_ptr<PoolCell>
    create(Network& net,
           const DeepNet& deepNet, 
           const std::string& name,
           const std::vector<unsigned int>& poolDims,
           unsigned int nbOutputs,
           const std::vector<unsigned int>& strideDims
              = std::vector<unsigned int>(2, 1U),
           const std::vector<unsigned int>& paddingDims
              = std::vector<unsigned int>(2, 0),
           Pooling pooling = Max,
           const std::shared_ptr<Activation>& /*activation*/
           = std::shared_ptr<Activation>())
    {
        return std::make_shared<PoolCell_Spike>(net,
                                                deepNet,
                                                name,
                                                poolDims,
                                                nbOutputs,
                                                strideDims,
                                                paddingDims,
                                                pooling);
    }

    virtual void setExtendedPadding(const std::vector<int>& paddingDims);
    virtual void initialize();
    virtual void
    propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void
    incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void notify(Time_T timestamp, NotifyType notify);
    virtual ~PoolCell_Spike() {};

protected:
    inline EventType_T maps(unsigned int output,
                            unsigned int ox,
                            unsigned int oy,
                            bool negative) const;
    inline std::tuple<unsigned int, unsigned int, unsigned int, bool>
    unmaps(EventType_T type) const;

    Tensor<int> mInputsActivity;
    // Coordinate of the input with max. activity for this output node
    Tensor<int> mInputMax;
    // Activity of the input with max. activity for this output node
    Tensor<int> mPoolActivity;
    // activity of the output node
    Tensor<int> mOutputsActivity;
    // mPoolNbChannels[output channel] -> number of input channels connected to
    // this output channel
    std::vector<unsigned int> mPoolNbChannels;

private:
    static Registrar<PoolCell> mRegistrar;
};
}

N2D2::EventType_T N2D2::PoolCell_Spike::maps(unsigned int output,
                                             unsigned int ox,
                                             unsigned int oy,
                                             bool negative) const
{
    if (output > 0x7FFF || ox > 0xFFFFFF || oy > 0xFFFFFF)
        throw std::domain_error("ConvCell::maps(): out of range");

    return (((EventType_T)output << 49) | ((EventType_T)ox << 25) | (oy << 1)
            | (int)negative);
}

std::tuple<unsigned int, unsigned int, unsigned int, bool>
N2D2::PoolCell_Spike::unmaps(EventType_T type) const
{
    return std::make_tuple<unsigned int, unsigned int, unsigned int, bool>(
        (type >> 49) & 0x7FFF,
        (type >> 25) & 0xFFFFFF,
        (type >> 1) & 0xFFFFFF,
        type & 1);
}

#endif // N2D2_POOLCELL_SPIKE_H
