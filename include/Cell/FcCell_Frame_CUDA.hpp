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
#include "Network.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T>
class FcCell_Frame_CUDA : public virtual FcCell, public Cell_Frame_CUDA<T> {
public:
    using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::mActivation;
    using Cell_Frame_CUDA<T>::mActivationDesc;

    FcCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                      unsigned int nbOutputs,
                      const std::shared_ptr<Activation>& activation
                      = std::make_shared
                      <TanhActivation_Frame_CUDA<T> >());
    static std::shared_ptr<FcCell>
    create(Network& /*net*/, const DeepNet& deepNet, 
           const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation>& activation
           = std::make_shared<TanhActivation_Frame_CUDA<T> >())
    {
        return std::make_shared<FcCell_Frame_CUDA>(deepNet, name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline void getWeight(unsigned int output, unsigned int channel,
                          BaseTensor& value) const;
    inline void getBias(unsigned int output, BaseTensor& value) const;
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
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
    
    std::pair<Float_T, Float_T> getFreeParametersRange(bool withAdditiveParameters = true) const;
    std::pair<Float_T, Float_T> getFreeParametersRangePerOutput(std::size_t output, 
                                                                bool withAdditiveParameters) const;
    
    void processFreeParameters(std::function<double(double)> func,
                               FreeParametersType type = All);
    void processFreeParametersPerOutput(std::function<double(double)> /*func*/,
                                        std::size_t /*output*/,
                                        FreeParametersType /*type*/ = All);
    
    virtual ~FcCell_Frame_CUDA();

protected:
    inline void setWeight(unsigned int output, unsigned int channel,
                          const BaseTensor& value);
    inline void setBias(unsigned int output, const BaseTensor& value);

    // Internal
    std::vector<std::shared_ptr<Solver> > mWeightsSolvers;
    CudaInterface<T> mSynapses;
    CudaTensor<T> mBias;
    CudaInterface<T> mDiffSynapses;
    CudaTensor<T> mDiffBias;

    T* mOnesVector; // Bias inputs
    mutable bool mSynchronized;

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

    if (!mSynchronized)
        mSynapses.synchronizeHToD(0, 0, channel, output, 1);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::getWeight(unsigned int output,
                                           unsigned int channel,
                                           BaseTensor& value) const
{
    if (!mSynchronized)
        mSynapses.synchronizeDToH(0, 0, channel, output, 1);

    value.resize({1});
    value = Tensor<T>({1}, mSynapses(0, 0, channel, output));
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::setBias(unsigned int output,
                                         const BaseTensor& value)
{
    mBias(output) = tensor_cast<T>(value)(0);

    if (!mSynchronized)
        mBias.synchronizeHToD(output, 1);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::getBias(unsigned int output,
                                         BaseTensor& value) const
{
    if (!mSynchronized)
        mBias.synchronizeDToH(output, 1);

    value.resize({1});
    value = Tensor<T>({1}, mBias(output));
}

namespace N2D2 {
template <> void FcCell_Frame_CUDA<half_float::half>::propagate(bool inference);
template <> void FcCell_Frame_CUDA<float>::propagate(bool inference);
template <> void FcCell_Frame_CUDA<double>::propagate(bool inference);

template <> void FcCell_Frame_CUDA<half_float::half>::backPropagate();
template <> void FcCell_Frame_CUDA<float>::backPropagate();
template <> void FcCell_Frame_CUDA<double>::backPropagate();
}

#endif // N2D2_FCCELL_FRAME_CUDA_H
