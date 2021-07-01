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

#ifndef N2D2_FCCELL_FRAME_CUDA_H
#define N2D2_FCCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "DeepNet.hpp"
#include "FcCell.hpp"
#include "Xnet/Network.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "containers/CudaTensor.hpp"


namespace N2D2 {
template <class T>
class FcCell_Frame_CUDA : public virtual FcCell, public Cell_Frame_CUDA<T> {
public:
    using Cell_Frame_CUDA<T>::keepInSync;
    using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::mActivation;
    using Cell_Frame_CUDA<T>::mActivationDesc;
    using Cell_Frame_CUDA<T>::mKeepInSync;
    using Cell_Frame_CUDA<T>::mDevices;

    FcCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                      unsigned int nbOutputs,
                      const std::shared_ptr<Activation>& activation
                      = std::shared_ptr<Activation>());
    static std::shared_ptr<FcCell>
    create(Network& /*net*/, const DeepNet& deepNet, 
           const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation>& activation
           = std::shared_ptr<Activation>())
    {
        return std::make_shared<FcCell_Frame_CUDA>(deepNet, name, nbOutputs, activation);
    }
    
    void resetWeights();
    void resetBias();

    virtual void initialize();
    virtual void initializeParameters(unsigned int nbInputChannels, unsigned int nbInputs);
    virtual void initializeWeightQuantizer();
    virtual void initializeDataDependent();
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline void getWeight(unsigned int output, unsigned int channel,
                          BaseTensor& value) const;
    inline void getQuantWeight(unsigned int output, unsigned int channel,
                          BaseTensor& value) const;
    inline void getBias(unsigned int output, BaseTensor& value) const;
    inline BaseInterface* getWeights()
    {
        return &mSynapses;
    };
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void logFreeParameters(const std::string& fileName,
                           unsigned int output) const;
    void logFreeParameters(const std::string& dirName) const;
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void exportFreeParameters(const std::string& fileName) const;
    void exportQuantFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName,
                              bool ignoreNotExists = false);
    void logFreeParametersDistrib(const std::string& fileName) const;
    void logQuantFreeParametersDistrib(const std::string& fileName) const;
    
    std::pair<Float_T, Float_T> getFreeParametersRange(FreeParametersType type = All) const;
    std::pair<Float_T, Float_T> getFreeParametersRangePerOutput(std::size_t output, 
                                                                FreeParametersType type = All) const;
    std::pair<Float_T, Float_T> getFreeParametersRangePerChannel(std::size_t channel) const;
    
    void processFreeParameters(std::function<Float_T(Float_T)> func,
                               FreeParametersType type = All);
    void processFreeParametersPerOutput(std::function<Float_T(Float_T)> /*func*/,
                                        std::size_t /*output*/,
                                        FreeParametersType /*type*/ = All);
    void processFreeParametersPerChannel(std::function<Float_T(Float_T)> /*func*/,
                                        std::size_t /*channel*/);
    
    void synchronizeToH(bool keepInSync_) const;
    void synchronizeToD(bool keepInSync_);
    virtual ~FcCell_Frame_CUDA();

protected:
    inline void setWeight(unsigned int output, unsigned int channel,
                          const BaseTensor& value);
    inline void setBias(unsigned int output, const BaseTensor& value);

    // Internal
    std::vector<std::shared_ptr<Solver> > mWeightsSolvers;
    CudaInterface<T> mSynapses;
    CudaTensor<T> mSynapsesNorm;
    CudaTensor<T> mBias;
    CudaInterface<T> mDiffSynapses;
    CudaTensor<T> mDiffBias;

    std::vector<T*> mOnesVector; // Bias inputs

private:
    static Registrar<FcCell> mRegistrar;
};
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::setWeight(unsigned int output,
                                           unsigned int channel,
                                           const BaseTensor& value)
{
    mSynapses(0, 0, channel, output) = tensor_cast<T>(value)(0);

    if (mKeepInSync)
        mSynapses.synchronizeHToD(0, 0, channel, output, 1);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::getWeight(unsigned int output,
                                           unsigned int channel,
                                           BaseTensor& value) const
{
    if (mKeepInSync)
        mSynapses.synchronizeDToH(0, 0, channel, output, 1);

    value.resize({1});
    value = Tensor<T>({1}, mSynapses(0, 0, channel, output));
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::getQuantWeight(unsigned int output,
                                           unsigned int channel,
                                           BaseTensor& value) const
{
    if (!mQuantizer)
        return;

    const CudaTensor<T>& synapses = cuda_tensor_cast<T>(mQuantizer->getQuantizedWeights(0));
    synapses.synchronizeDToH(0, 0, channel, output, 1);

    value.resize({1});
    value = Tensor<T>({1}, synapses(0, 0, channel, output));
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::setBias(unsigned int output,
                                         const BaseTensor& value)
{
    if (!mNoBias && mBias.empty())
        mBias.resize({getNbOutputs()});

    mBias(output) = tensor_cast<T>(value)(0);

    if (mKeepInSync)
        mBias.synchronizeHToD(output, 1);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::getBias(unsigned int output,
                                         BaseTensor& value) const
{
    if (mKeepInSync)
        mBias.synchronizeDToH(output, 1);

    value.resize({1});
    value = Tensor<T>({1}, mBias(output));
}

#endif // N2D2_FCCELL_FRAME_CUDA_H
