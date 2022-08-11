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
                        = std::shared_ptr<Activation>());
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
                    = std::shared_ptr<Activation>())
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

    void resetWeights();
    void resetBias();
    void resetWeightsSolver(const std::shared_ptr<Solver>& solver)
    {
        setWeightsSolver(solver);
        for (unsigned int k = 0, size = mWeightsSolvers.size(); k < size; ++k) {
            mWeightsSolvers[k] = mWeightsSolver->clone();
        }
    };

    virtual void setExtendedPadding(const std::vector<int>& paddingDims);
    /**
     * @brief Sets the Convolutional layer variables according to the given parameters and input features.
     */
    virtual void initialize();
    virtual void initializeParameters(unsigned int nbInputChannels, unsigned int nbInputs);
    virtual void initializeWeightQuantizer();
    /**
     * @brief Checks whether dimensions of a given input match those expected following the initialization.
     */
    virtual void check_input();
    virtual void initializeDataDependent();
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);

    /**
     * @brief Propagates input features through the convolutional layer and passes on processed features to the next layer.
     * 
     * @param inference False if the network is training. Input and ouput features are then saved for the backpropagation step.
     */
    virtual void propagate(bool inference = false);

    /**
     * @brief Computes gradient value for each parameter of the convolutional layer and passing on gradient to the next layers.
     * 
     * @param inference False if the network is training. Input and ouput features are then saved for the backpropagation step.
     */
    virtual void backPropagate();

    /**
     * @brief Updates weights and other parameters value following a given algorithm.
     */
    virtual void update();

    /**
     * @brief Fills a BaseTensor with the weight map connecting specific input and output channels.
     * 
     * @param output    Output channel index.
     * @param channel   Input channel index.
     * @param value     BaseTensor to be filled.
     */
    inline void getWeight(unsigned int output,
                          unsigned int channel,
                          BaseTensor& value) const
    {
        // const Tensor<T>& sharedSynapses
        //     = mSharedSynapses[mSharedSynapses.getTensorIndex(channel)];
        // channel -= mSharedSynapses.getTensorDataOffset(channel);

        // value.resize(sharedSynapses[output][channel].dims());
        // value = sharedSynapses[output][channel];

        unsigned int k = 0;
        unsigned int kChannelOffset = 0;

        for (; k < mSharedSynapses.size(); ++k) {
            const unsigned int kNbChannels = (mNbGroups[k] > 1)
                ? mSharedSynapses[k].dimZ() * mNbGroups[k]
                : mSharedSynapses[k].dimZ();

            if (channel < kChannelOffset + kNbChannels)
                break;
            else
                kChannelOffset += kNbChannels;
        }

        channel -= kChannelOffset;

        if (mNbGroups[k] > 1) {
            const size_t outputGroupSize = getNbOutputs() / mNbGroups[k];
            const size_t channelGroupSize = getNbChannels() / mNbGroups[k];

            const size_t outputGroup = output / outputGroupSize;
            const size_t channelGroup = channel / channelGroupSize;

            if (outputGroup != channelGroup) {
                const std::vector<size_t> kernelDims(mKernelDims.begin(),
                                                    mKernelDims.end());

                value.resize(kernelDims);
                value = Tensor<T>(kernelDims, T(0.0));
                return;
            }
            channel = channel % channelGroupSize;
        }

        const Tensor<T>& sharedSynapses = mSharedSynapses[k];
        value.resize(sharedSynapses[output][channel].dims());
        value = sharedSynapses[output][channel];
    };

    /**
     * @brief Get the Quantized Weight applied to a single input channel to compute a single ouput channel.
     * 
     * @param output Output channel index.
     * @param channel Input channel index.
     * @param value Tensor to be filled with quantized values.
     */
    inline void getQuantWeight(unsigned int output,
                          unsigned int channel,
                          BaseTensor& value) const
    {
        if (!mQuantizer)
            return;

        const Tensor<T>& sharedSynapses
            = tensor_cast<T>(mQuantizer->getQuantizedWeights(mSharedSynapses.getTensorIndex(channel)));
        channel -= mSharedSynapses.getTensorDataOffset(channel);

        value.resize(sharedSynapses[output][channel].dims());
        value = sharedSynapses[output][channel];

    };

    /**
     * @brief Fills a BaseTensor object with the bias values of the cell.
     * 
     * @param output    Output channel index of the cell.
     * @param value     BaseTensor to be filled.
     */
    inline void getBias(unsigned int output, BaseTensor& value) const
    {
        // Need to specify std::initializer_list<size_t> for GCC 4.4
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<T>({1}, (*mBias)(output));
    };

    /**
     * @brief Returns a reference to the weight interface of the layer.
     */
    inline BaseInterface* getWeights()
    {
        return &mSharedSynapses;
    };

    /**
     * @brief Returns a constant reference to the weight interface of the layer.
     */
    inline const BaseInterface* getWeights() const
    {
        return &mSharedSynapses;
    };

    void setWeights(unsigned int k,
                    BaseInterface* weights,
                    unsigned int offset);
    
    /**
     * @brief Returns a pointer to the bias tensor of the cell.
     */
    inline const std::shared_ptr<BaseTensor> getBiases() const
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
    /**
     * @brief Set the weight map connecting specific input channel and output according to the given value.
     * 
     * @param output    Output channel index.
     * @param channel   Inpt channel index.
     * @param value     Weight map.
     */
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          const BaseTensor& value)
    {
        // 1st: adjust channel value according to the number of groups
        unsigned int k = 0;
        unsigned int kChannelOffset = 0;

        for (; k < mSharedSynapses.size(); ++k) {
            const unsigned int kNbChannels = (mNbGroups[k] > 1)
                ? mSharedSynapses[k].dimZ() * mNbGroups[k]
                : mSharedSynapses[k].dimZ();

            if (channel < kChannelOffset + kNbChannels)
                break;
            else
                kChannelOffset += kNbChannels;
        }

        channel -= kChannelOffset;

        if (mNbGroups[k] > 1) {
            // assumption that all groups have the same size
            const size_t outputGroupSize = getNbOutputs() / mNbGroups[k];
            const size_t channelGroupSize = getNbChannels() / mNbGroups[k];
            // assumption that mapping is in the right order
            const size_t outputGroup = output / outputGroupSize;
            const size_t channelGroup = channel / channelGroupSize;

            if (outputGroup != channelGroup)
                return;

            channel = channel % channelGroupSize;
        }

        // 2nd: set weight value
        Tensor<T>& sharedSynapses = mSharedSynapses[k];

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
    std::vector<size_t> mNbGroups;
    std::vector<std::shared_ptr<Solver> > mWeightsSolvers;
    /// interface of input tensors expected by the cell for each individual synapse of a layer
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
