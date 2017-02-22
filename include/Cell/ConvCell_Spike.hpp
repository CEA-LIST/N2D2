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
                   unsigned int kernelWidth,
                   unsigned int kernelHeight,
                   unsigned int nbOutputs,
                   unsigned int subSampleX = 1,
                   unsigned int subSampleY = 1,
                   unsigned int strideX = 1,
                   unsigned int strideY = 1,
                   int paddingX = 0,
                   int paddingY = 0);
    static std::shared_ptr<ConvCell>
    create(Network& net,
           const std::string& name,
           unsigned int kernelWidth,
           unsigned int kernelHeight,
           unsigned int nbOutputs,
           unsigned int subSampleX = 1,
           unsigned int subSampleY = 1,
           unsigned int strideX = 1,
           unsigned int strideY = 1,
           int paddingX = 0,
           int paddingY = 0,
           const std::shared_ptr<Activation<Float_T> >& /*activation*/
           = std::shared_ptr<Activation<Float_T> >())
    {
        return std::make_shared<ConvCell_Spike>(net,
                                                name,
                                                kernelWidth,
                                                kernelHeight,
                                                nbOutputs,
                                                subSampleX,
                                                subSampleY,
                                                strideX,
                                                strideY,
                                                paddingX,
                                                paddingY);
    }

    virtual void initialize();
    virtual void
    propagateSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void
    incomingSpike(NodeIn* node, Time_T timestamp, EventType_T type = 0);
    virtual void notify(Time_T timestamp, NotifyType notify);
    inline Float_T getWeight(unsigned int output,
                             unsigned int channel,
                             unsigned int sx,
                             unsigned int sy) const;
    inline Float_T getBias(unsigned int /*output*/) const
    {
        return 0.0;
    };
    NodeOut*
    getOutput(unsigned int output, unsigned int ox, unsigned int oy) const
    {
        return mOutputs.at(output * (mOutputsWidth * mOutputsHeight)
                           + (ox + mOutputsWidth * oy));
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
                          unsigned int sx,
                          unsigned int sy,
                          Float_T value);
    inline void setBias(unsigned int /*output*/, Float_T /*value*/) {};
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
    Tensor4d<Synapse*> mSharedSynapses;

    Tensor4d<Time_T> mOutputsLastIntegration;
    Tensor4d<double> mOutputsIntegration;
    Tensor4d<Time_T> mOutputsRefractoryEnd;

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
                                     unsigned int sx,
                                     unsigned int sy,
                                     Float_T value)
{
    mSharedSynapses(sx, sy, channel, output)->setRelativeWeight(value);
}

N2D2::Float_T N2D2::ConvCell_Spike::getWeight(unsigned int output,
                                              unsigned int channel,
                                              unsigned int sx,
                                              unsigned int sy) const
{
    return mSharedSynapses(sx, sy, channel, output)->getRelativeWeight(true);
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
