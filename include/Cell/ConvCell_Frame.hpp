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

#ifndef N2D2_CONVCELL_FRAME_H
#define N2D2_CONVCELL_FRAME_H

#include "Cell_Frame.hpp"
#include "ConvCell.hpp"
#include "ConvCell_Frame_Kernels.hpp"
#include "DeepNet.hpp"
#include "Activation/TanhActivation_Frame.hpp"

namespace N2D2 {
template <class T>
class ConvCell_Frame : public virtual ConvCell, public Cell_Frame<T> {
public:
    using Cell_Frame<T>::mInputs;
    using Cell_Frame<T>::mOutputs;
    using Cell_Frame<T>::mDiffInputs;
    using Cell_Frame<T>::mDiffOutputs;

    ConvCell_Frame(const DeepNet& deepNet, const std::string& name,
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
                   const std::shared_ptr<Activation>& activation
                        = std::make_shared<TanhActivation_Frame<T> >());
    static std::shared_ptr<ConvCell> create(Network& /*net*/,
             const DeepNet& deepNet, 
             const std::string& name,
             const std::vector<unsigned int>& kernelDims,
             unsigned int nbOutputs,
             const std::vector<unsigned int>& subSampleDims
                    = std::vector<unsigned int>(2, 1U),
             const std::vector<unsigned int>& strideDims
                    = std::vector<unsigned int>(2, 1U),
             const std::vector<int>& paddingDims = std::vector<int>(2, 0),
             const std::vector<unsigned int>& dilationDims
                    = std::vector<unsigned int>(2, 1U),
             const std::shared_ptr<Activation>& activation
                    = std::make_shared<TanhActivation_Frame<T> >())
    {
        return std::make_shared<ConvCell_Frame<T> >(deepNet, 
                                                    name,
                                                    kernelDims,
                                                    nbOutputs,
                                                    subSampleDims,
                                                    strideDims,
                                                    paddingDims,
                                                    dilationDims,
                                                    activation);
    }

    virtual void setExtendedPadding(const std::vector<int>& paddingDims);
    virtual void initialize();
    virtual void initializeParameters(unsigned int inputDimZ, unsigned int nbInputs);
    virtual void initializeDataDependent();
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
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

        value.resize(sharedSynapses[output][channel].dims());
        value = sharedSynapses[output][channel];
    };
    inline void getQuantWeight(unsigned int /*output*/,
                          unsigned int /*channel*/,
                          BaseTensor& /*value*/) const
    {
        //nothing here for now
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
            throw std::runtime_error("ConvCell_Frame<T>::setBiases():"
                                     " invalid type");
        }
    }
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    virtual ~ConvCell_Frame();

protected:
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          const BaseTensor& value)
    {
        Tensor<T>& sharedSynapses
            = mSharedSynapses[mSharedSynapses.getTensorIndex(channel)];
        channel -= mSharedSynapses.getTensorDataOffset(channel);

        if (value.nbDims() < mKernelDims.size()) {
            for (size_t dim = 0; dim < value.nbDims(); ++dim) {
                assert(value.dims()[dim] == mKernelDims[dim]);
            }

            Tensor<T> valueND = tensor_cast<T>(value);
            valueND.reshape(std::vector<size_t>(mKernelDims.begin(),
                                                mKernelDims.end()));
            sharedSynapses[output][channel] = valueND;
        }
        else
            sharedSynapses[output][channel] = tensor_cast<T>(value);
    }
    inline void setBias(unsigned int output, const BaseTensor& value)
    {
        if (!mNoBias && mBias->empty())
            mBias->resize({1, 1, getNbOutputs(), 1});

        (*mBias)(output) = tensor_cast<T>(value)(0);
    };

    // Internal
    std::vector<std::shared_ptr<Solver> > mWeightsSolvers;
    Interface<T> mSharedSynapses;
    std::map<unsigned int,
        std::pair<Interface<T>*, unsigned int> > mExtSharedSynapses;
    std::shared_ptr<Tensor<T> > mBias;
    Interface<T> mDiffSharedSynapses;
    Tensor<T> mDiffBias;
    ConvCell_Frame_Kernels::Descriptor mConvDesc;

private:
    static Registrar<ConvCell> mRegistrar;
};
}

#endif // N2D2_CONVCELL_FRAME_H
