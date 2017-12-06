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
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
template <class T>
class RectifierActivation_Frame_CUDA : public RectifierActivation<T> {
public:
    static std::shared_ptr<RectifierActivation<T> > create()
    {
        return std::make_shared<RectifierActivation_Frame_CUDA<T> >();
    }

    RectifierActivation_Frame_CUDA();
    virtual void propagate(Tensor4d<T>* data);
    virtual void backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData);
    virtual ~RectifierActivation_Frame_CUDA();

    using RectifierActivation<T>::mLeakSlope;
    using RectifierActivation<T>::mShifting;
    using RectifierActivation<T>::mClipping;

protected:
#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t mActivationDesc;
#else
    cudnnActivationMode_t mActivationDesc;
#endif

private:
    static Registrar<RectifierActivation<T> > mRegistrar;
};
}

template <class T>
N2D2::RectifierActivation_Frame_CUDA<T>::RectifierActivation_Frame_CUDA():
    RectifierActivation<T>()
{
#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&mActivationDesc));
    CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor(
        mActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
#else
    mActivationDesc = CUDNN_ACTIVATION_RELU;
#endif
}

namespace N2D2 {
template <>
void RectifierActivation_Frame_CUDA<float>::propagate(Tensor4d<float>* data);
template <>
void RectifierActivation_Frame_CUDA
    <float>::backPropagate(Tensor4d<float>* data, Tensor4d<float>* diffData);

template <>
void RectifierActivation_Frame_CUDA<double>::propagate(Tensor4d<double>* data);
template <>
void RectifierActivation_Frame_CUDA
    <double>::backPropagate(Tensor4d<double>* data, Tensor4d<double>* diffData);
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
