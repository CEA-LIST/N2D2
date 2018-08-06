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

#ifndef N2D2_RECTIFIERACTIVATION_FRAME_CUDA_H
#define N2D2_RECTIFIERACTIVATION_FRAME_CUDA_H

#include "Activation/Activation_CUDA_Kernels.hpp"
#include "Activation/RectifierActivation.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T>
class RectifierActivation_Frame_CUDA : public RectifierActivation {
public:
    static std::shared_ptr<RectifierActivation> create()
    {
        return std::make_shared<RectifierActivation_Frame_CUDA<T> >();
    }

    RectifierActivation_Frame_CUDA();
    inline virtual void propagate(BaseTensor& data);
    inline virtual void backPropagate(BaseTensor& data, BaseTensor& diffData);
    void propagate(CudaTensor<T>& data);
    void backPropagate(CudaTensor<T>& data, CudaTensor<T>& diffData);
    virtual ~RectifierActivation_Frame_CUDA();

protected:
#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t mActivationDesc;
#else
    cudnnActivationMode_t mActivationDesc;
#endif

private:
    static Registrar<RectifierActivation > mRegistrar;
};
}

template <class T>
N2D2::RectifierActivation_Frame_CUDA<T>::RectifierActivation_Frame_CUDA():
    RectifierActivation()
{
#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&mActivationDesc));
    CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor(
        mActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
#else
    mActivationDesc = CUDNN_ACTIVATION_RELU;
#endif
}

template <class T>
void N2D2::RectifierActivation_Frame_CUDA<T>::propagate(BaseTensor& data) {
    propagate(dynamic_cast<CudaTensor<T>&>(data));
}

template <class T>
void N2D2::RectifierActivation_Frame_CUDA<T>::backPropagate(BaseTensor& data,
                                                        BaseTensor& diffData) {
    backPropagate(dynamic_cast<CudaTensor<T>&>(data),
                  dynamic_cast<CudaTensor<T>&>(diffData));
}

namespace N2D2 {
template <>
void RectifierActivation_Frame_CUDA<half_float::half>::propagate(CudaTensor<half_float::half>& data);
template <>
void RectifierActivation_Frame_CUDA
    <half_float::half>::backPropagate(CudaTensor<half_float::half>& data, CudaTensor<half_float::half>& diffData);

template <>
void RectifierActivation_Frame_CUDA<float>::propagate(CudaTensor<float>& data);
template <>
void RectifierActivation_Frame_CUDA
    <float>::backPropagate(CudaTensor<float>& data, CudaTensor<float>& diffData);

template <>
void RectifierActivation_Frame_CUDA<double>::propagate(CudaTensor<double>& data);
template <>
void RectifierActivation_Frame_CUDA
    <double>::backPropagate(CudaTensor<double>& data, CudaTensor<double>& diffData);
}

template <class T>
N2D2::RectifierActivation_Frame_CUDA<T>::~RectifierActivation_Frame_CUDA()
{
// dtor
#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnDestroyActivationDescriptor(mActivationDesc));
#endif
}

#endif // N2D2_RECTIFIERACTIVATION_FRAME_CUDA_H
