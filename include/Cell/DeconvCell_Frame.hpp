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

#ifndef N2D2_DECONVCELL_FRAME_H
#define N2D2_DECONVCELL_FRAME_H

#include "Cell_Frame.hpp"
#include "ConvCell_Frame_Kernels.hpp"
#include "DeconvCell.hpp"
#include "DeepNet.hpp"
#include "Activation/TanhActivation_Frame.hpp"

namespace N2D2 {
template <class T>
class DeconvCell_Frame : public virtual DeconvCell, public Cell_Frame<T> {
public:
    using Cell_Frame<T>::mInputs;
    using Cell_Frame<T>::mOutputs;
    using Cell_Frame<T>::mDiffInputs;
    using Cell_Frame<T>::mDiffOutputs;

    DeconvCell_Frame(const DeepNet& deepNet, const std::string& name,
                     const std::vector<unsigned int>& kernelDims,
                     unsigned int nbOutputs,
                     const std::vector<unsigned int>& strideDims
                          = std::vector<unsigned int>(2, 1U),
                     const std::vector<int>& paddingDims
                        = std::vector<int>(2, 0),
                     const std::vector<unsigned int>& dilationDims
                          = std::vector<unsigned int>(2, 1U),
                       const std::shared_ptr<Activation>& activation
                     = std::shared_ptr<Activation>());
    static std::shared_ptr<DeconvCell>
    create(Network& /*net*/, const DeepNet& deepNet, 
           const std::string& name,
           const std::vector<unsigned int>& kernelDims,
           unsigned int nbOutputs,
           const std::vector<unsigned int>& strideDims
                  = std::vector<unsigned int>(2, 1U),
           const std::vector<int>& paddingDims = std::vector<int>(2, 0),
           const std::vector<unsigned int>& dilationDims
                  = std::vector<unsigned int>(2, 1U),
           const std::shared_ptr<Activation>& activation
           = std::shared_ptr<Activation>())
    {
        return std::make_shared<DeconvCell_Frame<T> >(deepNet, name, 
                                                      kernelDims,
                                                      nbOutputs,
                                                      strideDims,
                                                      paddingDims,
                                                      dilationDims,
                                                      activation);
    }

    virtual void initialize();
    virtual void initializeParameters(unsigned int nbInputChannels, unsigned int nbInputs);
    virtual void check_input();
    virtual void initializeDataDependent();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline void getWeight(unsigned int output,
                          unsigned int channel,
                          BaseTensor& value) const
    {
        const Tensor<T>& sharedSynapses
            = mSharedSynapses[mSharedSynapses.getTensorIndex(channel)];
        channel -= mSharedSynapses.getTensorDataOffset(channel);

        value.resize(sharedSynapses[channel][output].dims());
        value = sharedSynapses[channel][output];
    };
    inline void getBias(unsigned int output, BaseTensor& value) const
    {
        // Need to specify std::initializer_list<size_t> for GCC 4.4
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<T>({1}, (*mBias)(output));
    };
    inline BaseInterface* getWeights()
    {
        return &mSharedSynapses;
    };
    void setWeights(unsigned int k,
                    BaseInterface* weights,
                    unsigned int offset);
    inline std::shared_ptr<BaseTensor> getBiases()
    {
        return mBias;
    };
    inline void setBiases(const std::shared_ptr<BaseTensor>& biases)
    {
        mBias = std::dynamic_pointer_cast<Tensor<T> >(biases);

        if (!mBias) {
            throw std::runtime_error("DeconvCell_Frame<T>::setBiases():"
                                     " invalid type");
        }
    }
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    virtual ~DeconvCell_Frame();

protected:
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          const BaseTensor& value)
    {
        Tensor<T>& sharedSynapses
            = mSharedSynapses[mSharedSynapses.getTensorIndex(channel)];
        channel -= mSharedSynapses.getTensorDataOffset(channel);

        sharedSynapses[channel][output] = tensor_cast<T>(value);
    }
    inline void setBias(unsigned int output, const BaseTensor& value)
    {
        (*mBias)(output) = tensor_cast<T>(value)(0);
    };

    // Internal
    std::vector<std::shared_ptr<Solver> > mWeightsSolvers;
    Interface<T,-1> mSharedSynapses;
    std::map<unsigned int,
        std::pair<Interface<T>*, unsigned int> > mExtSharedSynapses;
    std::shared_ptr<Tensor<T> > mBias;
    Interface<T,-1> mDiffSharedSynapses;
    Tensor<T> mDiffBias;
    ConvCell_Frame_Kernels::Descriptor mConvDesc;

private:
    static Registrar<DeconvCell> mRegistrar;
};
}

#endif // N2D2_DECONVCELL_FRAME_H
