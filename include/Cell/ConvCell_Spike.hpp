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

#ifndef N2D2_CONVCELL_SPIKE_H
#define N2D2_CONVCELL_SPIKE_H

#include "Xcell.hpp"

#include "Cell_Spike.hpp"
#include "ConvCell.hpp"
#include "NodeIn.hpp"
#include "NodeOut.hpp"

namespace N2D2 {
class ConvCell_Spike : public virtual ConvCell, public Cell_Spike {
public:
    ConvCell_Spike(Network& net,
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
    create(Network& net,
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
        return std::make_shared<ConvCell_Spike>(net,
                                                name,
                                                kernelDims,
                                                nbOutputs,
                                                subSampleDims,
                                                strideDims,
                                                paddingDims,
                                                dilationDims);
    }

    virtual void initialize();
    virtual void
    propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void
    incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void notify(Time_T timestamp, NotifyType notify);
    inline void getWeight(unsigned int output,
                          unsigned int channel,
                          BaseTensor& value) const;
    inline void getBias(unsigned int /*output*/, BaseTensor& value) const
    {
        value.resize({1});
        value = Tensor<Float_T>({1}, 0.0);
    };
    NodeOut*
    getOutput(unsigned int output, unsigned int ox, unsigned int oy) const
    {
        return mOutputs.at(output * (mOutputsDims[0] * mOutputsDims[1])
                           + (ox + mOutputsDims[0] * oy));
    }
    cv::Mat reconstructActivity(unsigned int output,
                                Time_T start,
                                Time_T stop,
                                bool normalize = false) const;
    void reconstructActivities(const std::string& dirName,
                               Time_T start,
                               Time_T stop,
                               bool normalize = false) const;
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    Synapse::Stats logStats(const std::string& dirName) const;
    virtual ~ConvCell_Spike();

protected:
    virtual Synapse* newSynapse() const;
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          const BaseTensor& value);
    inline void setBias(unsigned int /*output*/, const BaseTensor& /*value*/) {};
    inline EventType_T maps(unsigned int output,
                            unsigned int ox,
                            unsigned int oy,
                            bool negative) const;
    inline std::tuple<unsigned int, unsigned int, unsigned int, bool>
    unmaps(EventType_T type) const;

    /// Relative initial synaptic weight \f$w_{init}\f$
    ParameterWithSpread<Weight_T> mWeightsRelInit;

    /// Threshold of the neuron \f$I_{thres}\f$
    Parameter<double> mThreshold;
    Parameter<bool> mBipolarThreshold;
    /// Neural leak time constant \f$\tau_{leak}\f$ (if 0, no leak)
    Parameter<Time_T> mLeak;
    /// Neural refractory period \f$T_{refrac}\f$
    Parameter<Time_T> mRefractory;

    // mSharedSynapses[output feature map][input channel][synapse, in a 2D
    // matrix = convolution kernel]
    Tensor<Synapse*> mSharedSynapses;

    Tensor<Time_T> mOutputsLastIntegration;
    Tensor<double> mOutputsIntegration;
    Tensor<Time_T> mOutputsRefractoryEnd;

private:
    static Registrar<ConvCell> mRegistrar;
};

void addInput(Xcell& cell,
              ConvCell_Spike& convCell,
              unsigned int output,
              unsigned int x0 = 0,
              unsigned int y0 = 0,
              unsigned int width = 0,
              unsigned int height = 0);
void addInput(Xcell& cell, ConvCell_Spike& convCell);
}

void N2D2::ConvCell_Spike::setWeight(unsigned int output,
                                     unsigned int channel,
                                     const BaseTensor& value)
{
    Tensor<Synapse*> sharedSynapses = mSharedSynapses[output][channel];
    assert(value.dims() == sharedSynapses.dims());

    const Tensor<Float_T>& kernel = tensor_cast<Float_T>(value);

    for (size_t index = 0; index < value.size(); ++index)
        sharedSynapses(index)->setRelativeWeight(kernel(index));
}

void N2D2::ConvCell_Spike::getWeight(unsigned int output,
                                     unsigned int channel,
                                     BaseTensor& value) const
{
    const Tensor<Synapse*>& sharedSynapses = mSharedSynapses[output][channel];
    Tensor<Float_T> values(sharedSynapses.dims());

    for (size_t index = 0; index < values.size(); ++index)
        values(index) = sharedSynapses(index)->getRelativeWeight(true);

    value.resize(values.dims());
    value = values;
}

N2D2::EventType_T N2D2::ConvCell_Spike::maps(unsigned int output,
                                             unsigned int ox,
                                             unsigned int oy,
                                             bool negative) const
{
    if (output > 0x7FFF || ox > 0xFFFFFF || oy > 0xFFFFFF)
        throw std::domain_error("ConvCell_Spike::maps(): out of range");

    return (((EventType_T)output << 49) | ((EventType_T)ox << 25) | (oy << 1)
            | (int)negative);
}

std::tuple<unsigned int, unsigned int, unsigned int, bool>
N2D2::ConvCell_Spike::unmaps(EventType_T type) const
{
    return std::make_tuple<unsigned int, unsigned int, unsigned int, bool>(
        (type >> 49) & 0x7FFF,
        (type >> 25) & 0xFFFFFF,
        (type >> 1) & 0xFFFFFF,
        type & 1);
}

#endif // N2D2_CONVCELL_SPIKE_H
