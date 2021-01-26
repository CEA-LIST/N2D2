/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_LOGISTICACTIVATION_FRAME_CUDA_H
#define N2D2_LOGISTICACTIVATION_FRAME_CUDA_H

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "Activation/Activation_CUDA_Kernels.hpp"
#include "Activation/LogisticActivation.hpp"
#include "Cell/Cell.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T>
class LogisticActivation_Frame_CUDA : public LogisticActivation {
public:
    static std::shared_ptr<LogisticActivation> create(bool withLoss = false)
    {
        return std::make_shared<LogisticActivation_Frame_CUDA<T> >(withLoss);
    }

    LogisticActivation_Frame_CUDA(bool withLoss = false);
    virtual void propagate(const Cell& cell,
                           const BaseTensor& input,
                           BaseTensor& output,
                           bool inference = false);
    virtual void backPropagate(const Cell& cell,
                               const BaseTensor& input,
                               const BaseTensor& output,
                               const BaseTensor& diffInput,
                               BaseTensor& diffOutput);
    virtual void update(unsigned int batchSize);
    virtual ~LogisticActivation_Frame_CUDA();

protected:
#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t mActivationDesc;
#else
    cudnnActivationMode_t mActivationDesc;
#endif

private:
    static Registrar<LogisticActivation> mRegistrar;
};
}

template <class T>
N2D2::LogisticActivation_Frame_CUDA<T>::LogisticActivation_Frame_CUDA(bool withLoss)
    : LogisticActivation(withLoss)
{
#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&mActivationDesc));
    CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor(mActivationDesc,
                                                    CUDNN_ACTIVATION_SIGMOID,
                                                    CUDNN_NOT_PROPAGATE_NAN,
                                                    0.0));
#else
    mActivationDesc = CUDNN_ACTIVATION_SIGMOID;
#endif
}

template <class T>
void N2D2::LogisticActivation_Frame_CUDA<T>::propagate(
    const Cell& cell, 
    const BaseTensor& baseInput,
    BaseTensor& baseOutput,
    bool /*inference*/)
{
    const CudaTensor<T>& input = dynamic_cast<const CudaTensor<T>&>(baseInput);
    CudaTensor<T>& output = dynamic_cast<CudaTensor<T>&>(baseOutput);

    if (!LogisticActivationDisabled) {
        mScaling.propagate(cell, input, output);

        const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;
        const typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

        CHECK_CUDNN_STATUS(cudnnActivationForward(CudaContext::cudnnHandle(),
                                                  mActivationDesc,
                                                  &alpha,
                                                  output.getCudnnTensorDesc(),
                                                  output.getDevicePtr(),
                                                  &beta,
                                                  output.getCudnnTensorDesc(),
                                                  output.getDevicePtr()));
    }
}

template <class T>
void N2D2::LogisticActivation_Frame_CUDA<T>::backPropagate(
    const Cell& cell, 
    const BaseTensor& /*baseInput*/,
    const BaseTensor& baseOutput,
    const BaseTensor& baseDiffInput,
    BaseTensor& baseDiffOutput)
{
    if (LogisticActivationDisabled)
        return;

    const CudaTensor<T>& output = dynamic_cast<const CudaTensor<T>&>(baseOutput);
    const CudaTensor<T>& diffInput = dynamic_cast<const CudaTensor<T>&>(baseDiffInput);
    CudaTensor<T>& diffOutput = dynamic_cast<CudaTensor<T>&>(baseDiffOutput);

    if (!this->mWithLoss) {
        const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;
        const typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

        CHECK_CUDNN_STATUS(
            cudnnActivationBackward(CudaContext::cudnnHandle(),
                                    mActivationDesc,
                                    &alpha,
                                    output.getCudnnTensorDesc(),
                                    output.getDevicePtr(),
                                    diffInput.getCudnnTensorDesc(),
                                    diffInput.getDevicePtr(),
                                    output.getCudnnTensorDesc(),
                                    output.getDevicePtr(),
                                    &beta,
                                    diffOutput.getCudnnTensorDesc(),
                                    diffOutput.getDevicePtr()));

        mScaling.backPropagate(cell, diffOutput, diffOutput);
    }
    else
        mScaling.backPropagate(cell, diffInput, diffOutput);
}

template <class T>
N2D2::LogisticActivation_Frame_CUDA<T>::~LogisticActivation_Frame_CUDA()
{
// dtor
#if CUDNN_VERSION >= 5000
    cudnnDestroyActivationDescriptor(mActivationDesc);
#endif
}
template <class T>
void N2D2::LogisticActivation_Frame_CUDA<T>::update(unsigned int batchSize)
{
    /*
    if(mQuantizer) {
        mQuantizer->update(batchSize);
    }
    */
}
#endif // N2D2_LOGISTICACTIVATION_FRAME_CUDA_H
