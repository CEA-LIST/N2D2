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

#ifndef N2D2_TANHACTIVATION_FRAME_CUDA_H
#define N2D2_TANHACTIVATION_FRAME_CUDA_H

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "Activation/Activation_Kernels.hpp"
#include "Activation/Activation_CUDA_Kernels.hpp"
#include "Activation/TanhActivation.hpp"
#include "Cell/Cell.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T> 
class TanhActivation_Frame_CUDA : public TanhActivation {
public:
    static std::shared_ptr<TanhActivation> create()
    {
        return std::make_shared<TanhActivation_Frame_CUDA<T> >();
    }

    TanhActivation_Frame_CUDA();

    virtual void propagate(const Cell& cell, BaseTensor& data, bool inference = false);
    virtual void backPropagate(const Cell& cell, BaseTensor& data, BaseTensor& diffData);

    void propagate(const Cell& cell, CudaTensor<T>& data, bool inference = false);

    virtual ~TanhActivation_Frame_CUDA();

protected:
#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t mActivationDesc;
#else
    cudnnActivationMode_t mActivationDesc;
#endif

private:
    static Registrar<TanhActivation> mRegistrar;
};
}

template <class T>
N2D2::TanhActivation_Frame_CUDA<T>::TanhActivation_Frame_CUDA()
{
#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&mActivationDesc));
    CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor(
        mActivationDesc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0));
#else
    mActivationDesc = CUDNN_ACTIVATION_TANH;
#endif
}

template <class T>
void N2D2::TanhActivation_Frame_CUDA<T>::propagate(const Cell& cell, 
                                                   BaseTensor& data, bool inference)
{
    CudaTensor<T>& cudaData = dynamic_cast<CudaTensor<T>&>(data);

    mScaling.propagate(cell, cudaData);

    const typename Cuda::cudnn_scaling_type<T>::type alpha = mAlpha;
    const typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

    CHECK_CUDNN_STATUS(cudnnActivationForward(CudaContext::cudnnHandle(),
                                              mActivationDesc,
                                              &alpha,
                                              cudaData.getCudnnTensorDesc(),
                                              cudaData.getDevicePtr(),
                                              &beta,
                                              cudaData.getCudnnTensorDesc(),
                                              cudaData.getDevicePtr()));

    propagate(cell, cudaData, inference);
}

namespace N2D2 {
template <>
void TanhActivation_Frame_CUDA<half_float::half>::propagate(const Cell& cell, 
                                                            CudaTensor<half_float::half>& data, 
                                                            bool inference);

template <>
void TanhActivation_Frame_CUDA<float>::propagate(const Cell& cell, 
                                                 CudaTensor<float>& data, bool inference);

template <>
void TanhActivation_Frame_CUDA<double>::propagate(const Cell& cell, 
                                                  CudaTensor<double>& data, bool inference);
}

template <class T>
void N2D2::TanhActivation_Frame_CUDA<T>::backPropagate(const Cell& cell, 
                                                       BaseTensor& data, BaseTensor& diffData)
{
    CudaTensor<T>& cudaData = dynamic_cast<CudaTensor<T>&>(data);
    CudaTensor<T>& cudaDiffData = dynamic_cast<CudaTensor<T>&>(diffData);

    const typename Cuda::cudnn_scaling_type<T>::type alpha = mAlpha;
    const typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

    CHECK_CUDNN_STATUS(
        cudnnActivationBackward(CudaContext::cudnnHandle(),
                                mActivationDesc,
                                &alpha,
                                cudaData.getCudnnTensorDesc(),
                                cudaData.getDevicePtr(),
                                cudaDiffData.getCudnnTensorDesc(),
                                cudaDiffData.getDevicePtr(),
                                cudaData.getCudnnTensorDesc(),
                                cudaData.getDevicePtr(),
                                &beta,
                                cudaDiffData.getCudnnTensorDesc(),
                                cudaDiffData.getDevicePtr()));
    
    mScaling.backPropagate(cell, cudaData, cudaDiffData);
}

template <class T>
N2D2::TanhActivation_Frame_CUDA<T>::~TanhActivation_Frame_CUDA()
{
// dtor
#if CUDNN_VERSION >= 5000
    cudnnDestroyActivationDescriptor(mActivationDesc);
#endif
}

#endif // N2D2_TANHACTIVATION_FRAME_CUDA_H
