/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#ifndef N2D2_CONVCELL_FRAME_CUDA_H
#define N2D2_CONVCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "ConvCell.hpp"
#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "DeepNet.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T>
class ConvCell_Frame_CUDA : public virtual ConvCell, public Cell_Frame_CUDA<T> {
public:
    using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::mActivation;
    using Cell_Frame_CUDA<T>::mActivationDesc;

    ConvCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
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
                    = std::make_shared<TanhActivation_Frame_CUDA<T> >());
    static std::shared_ptr<ConvCell>
    create(Network& /*net*/,
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
           = std::make_shared<TanhActivation_Frame_CUDA<T> >())
    {
        return std::make_shared<ConvCell_Frame_CUDA<T> >(deepNet,
                                                         name,
                                                         kernelDims,
                                                         nbOutputs,
                                                         subSampleDims,
                                                         strideDims,
                                                         paddingDims,
                                                         dilationDims,
                                                         activation);
    }

    virtual void initialize();
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline void getWeight(unsigned int output,
                          unsigned int channel,
                          BaseTensor& value) const;
    inline void getBias(unsigned int output, BaseTensor& value) const;
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
    void setBiases(const std::shared_ptr<BaseTensor>& biases);
    void checkGradient(double /*epsilon*/ = 1.0e-4,
                       double /*maxError*/ = 1.0e-6);
    void logFreeParameters(const std::string& fileName,
                           unsigned int output,
                           unsigned int channel) const;
    void logFreeParameters(const std::string& fileName,
                           unsigned int output) const;
    void logFreeParameters(const std::string& dirName) const;
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName,
                              bool ignoreNotExists = false);
    void logFreeParametersDistrib(const std::string& fileName) const;
    void discretizeFreeParameters(unsigned int nbLevels);
    std::pair<Float_T, Float_T> getFreeParametersRange() const;
    void processFreeParameters(const std::function
                               <double(const double&)>& func,
                               FreeParametersType type = All);
    virtual ~ConvCell_Frame_CUDA();

protected:
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          const BaseTensor& value);
    inline void setBias(unsigned int output, const BaseTensor& value);

    // Internal
    std::vector<size_t> mNbGroups;
    std::vector<std::shared_ptr<Solver> > mWeightsSolvers;
    CudaInterface<T> mSharedSynapses;
    std::map<unsigned int,
        std::pair<CudaInterface<T>*, unsigned int> > mExtSharedSynapses;
    std::shared_ptr<CudaTensor<T> > mBias;
    CudaInterface<T> mDiffSharedSynapses;
    CudaTensor<T> mDiffBias;

    size_t mWorkspaceSize;
    void* mWorkspace;

    std::vector<cudnnFilterDescriptor_t> mFilterDesc;
    std::vector<cudnnConvolutionFwdAlgo_t> mFwdAlgo;
#if CUDNN_VERSION >= 5000
    std::vector<cudnnConvolutionBwdFilterAlgo_t> mBwdFilterAlgo;
    std::vector<cudnnConvolutionBwdDataAlgo_t> mBwdDataAlgo;
#endif

    cudnnConvolutionDescriptor_t mConvDesc;
    mutable bool mSynchronized;

private:
    static Registrar<ConvCell> mRegistrar;
};
}

template <class T>
void N2D2::ConvCell_Frame_CUDA<T>::setWeight(unsigned int output,
                                             unsigned int channel,
                                             const BaseTensor& value)
{
    const unsigned int k = mInputs.getTensorIndex(channel);
    channel -= mInputs.getTensorDataOffset(channel);

#if CUDNN_VERSION >= 7000
    if (mNbGroups[k] > 1) {
        const size_t outputGroupSize = getNbOutputs() / mNbGroups[k];
        const size_t channelGroupSize = mInputs[k].dimZ() / mNbGroups[k];
        const size_t outputGroup = output / outputGroupSize;
        const size_t channelGroup = channel / channelGroupSize;

        if (outputGroup != channelGroup)
            return;

        channel = channel % channelGroupSize;
    }
#endif

    CudaTensor<T>& sharedSynapses = mSharedSynapses[k];
    sharedSynapses[output][channel] = tensor_cast<T>(value);

    if (!mSynchronized)
        sharedSynapses[output][channel].synchronizeHToD();
}

template <class T>
void
N2D2::ConvCell_Frame_CUDA<T>::getWeight(unsigned int output,
                                        unsigned int channel,
                                        BaseTensor& value) const
{
    const unsigned int k = mInputs.getTensorIndex(channel);
    channel -= mInputs.getTensorDataOffset(channel);

#if CUDNN_VERSION >= 7000
    if (mNbGroups[k] > 1) {
        const size_t outputGroupSize = getNbOutputs() / mNbGroups[k];
        const size_t channelGroupSize = mInputs[k].dimZ() / mNbGroups[k];
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
#endif

    const CudaTensor<T>& sharedSynapses = mSharedSynapses[k];

    if (!mSynchronized)
        sharedSynapses[output][channel].synchronizeDToH();

    value.resize(sharedSynapses[output][channel].dims());
    value = sharedSynapses[output][channel];
}

template <class T>
void N2D2::ConvCell_Frame_CUDA<T>::setBias(unsigned int output,
                                           const BaseTensor& value)
{
    if (!mNoBias && mBias->empty())
        mBias->resize({1, 1, getNbOutputs(), 1});

    (*mBias)(output) = tensor_cast<T>(value)(0);

    if (!mSynchronized)
        mBias->synchronizeHToD(output, 1);
}

template <class T>
void N2D2::ConvCell_Frame_CUDA<T>::getBias(unsigned int output,
                                           BaseTensor& value) const
{
    if (!mSynchronized)
        mBias->synchronizeDToH(output, 1);

    value.resize({1});
    value = Tensor<T>({1}, (*mBias)(output));
}

#endif // N2D2_CONVCELL_FRAME_CUDA_H
