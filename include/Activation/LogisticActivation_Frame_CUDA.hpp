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

#include "Activation/LogisticActivation.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
template <class T>
class LogisticActivation_Frame_CUDA : public LogisticActivation<T> {
public:
    static std::shared_ptr<LogisticActivation<T> > create(bool withLoss = false)
    {
        return std::make_shared<LogisticActivation_Frame_CUDA<T> >(withLoss);
    }

    LogisticActivation_Frame_CUDA(bool withLoss = false);
    virtual void propagate(Tensor4d<T>* data);
    virtual void backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData);
    virtual ~LogisticActivation_Frame_CUDA();

protected:
#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t mActivationDesc;
#else
    cudnnActivationMode_t mActivationDesc;
#endif

private:
    static Registrar<LogisticActivation<T> > mRegistrar;
};
}

template <class T>
N2D2::LogisticActivation_Frame_CUDA
    <T>::LogisticActivation_Frame_CUDA(bool withLoss)
    : LogisticActivation<T>(withLoss)
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
void N2D2::LogisticActivation_Frame_CUDA<T>::propagate(Tensor4d<T>* data)
{
    if (LogisticActivationDisabled)
        return;

    CudaTensor4d<T>* cudaData = static_cast<CudaTensor4d<T>*>(data);

    const T alpha = 1.0f;
    const T beta = 0.0f;

    CHECK_CUDNN_STATUS(cudnnActivationForward(CudaContext::cudnnHandle(),
                                              mActivationDesc,
                                              &alpha,
                                              cudaData->getCudnnTensorDesc(),
                                              cudaData->getDevicePtr(),
                                              &beta,
                                              cudaData->getCudnnTensorDesc(),
                                              cudaData->getDevicePtr()));
}

template <class T>
void N2D2::LogisticActivation_Frame_CUDA
    <T>::backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData)
{
    if (LogisticActivationDisabled)
        return;

    if (!this->mWithLoss) {
        CudaTensor4d<T>* cudaData = static_cast<CudaTensor4d<T>*>(data);
        CudaTensor4d<T>* cudaDiffData = static_cast<CudaTensor4d<T>*>(diffData);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CHECK_CUDNN_STATUS(
            cudnnActivationBackward(CudaContext::cudnnHandle(),
                                    mActivationDesc,
                                    &alpha,
                                    cudaData->getCudnnTensorDesc(),
                                    cudaData->getDevicePtr(),
                                    cudaDiffData->getCudnnTensorDesc(),
                                    cudaDiffData->getDevicePtr(),
                                    cudaData->getCudnnTensorDesc(),
                                    cudaData->getDevicePtr(),
                                    &beta,
                                    cudaDiffData->getCudnnTensorDesc(),
                                    cudaDiffData->getDevicePtr()));
    }
}

template <class T>
N2D2::LogisticActivation_Frame_CUDA<T>::~LogisticActivation_Frame_CUDA()
{
// dtor
#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnDestroyActivationDescriptor(mActivationDesc));
#endif
}

#endif // N2D2_LOGISTICACTIVATION_FRAME_CUDA_H
