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

#ifdef CUDA
#include <cudnn.h>
#if CUDNN_VERSION >= 3000

#include "Cell/LRNCell_Frame_CUDA.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
    N2D2::LRNCell_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::LRNCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
    N2D2::LRNCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::LRNCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::LRNCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::LRNCell>::Type<double>());

template <class T>
N2D2::LRNCell_Frame_CUDA<T>::LRNCell_Frame_CUDA(const std::string& name,
                                             unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      LRNCell(name, nbOutputs),
      Cell_Frame_CUDA<T>(name, nbOutputs)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateLRNDescriptor(&mLRNDesc));
}

template <class T>
void N2D2::LRNCell_Frame_CUDA<T>::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("LRNCell_Frame_CUDA<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    CHECK_CUDNN_STATUS(cudnnSetLRNDescriptor(mLRNDesc, mN, mAlpha, mBeta, mK));

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for LRNCell " + mName);

        mOutputDesc.push_back(cudnnTensorDescriptor_t());

        CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mOutputDesc.back()));
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptorEx(
            mOutputDesc.back(),
            CudaContext::data_type<T>::value,
            mOutputs.dimB(),
            mInputs[k].dimZ(),
            mOutputs.dimY(),
            mOutputs.dimX(),
            mOutputs.dimX() * mOutputs.dimY() * mInputs.dimZ(),
            mOutputs.dimX() * mOutputs.dimY(),
            mOutputs.dimX(),
            1));
    }
}

template <class T>
void N2D2::LRNCell_Frame_CUDA<T>::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;
    const typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast<T>(mInputs[k]);

        CHECK_CUDNN_STATUS(
            cudnnLRNCrossChannelForward(CudaContext::cudnnHandle(),
                                        mLRNDesc,
                                        CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                        &alpha,
                                        input->getCudnnTensorDesc(),
                                        input->getDevicePtr(),
                                        &beta,
                                        mOutputDesc[k],
                                        mOutputs.getDevicePtr() + offset));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
    }

    mDiffInputs.clearValid();
}

template <class T>
void N2D2::LRNCell_Frame_CUDA<T>::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const typename Cuda::cudnn_scaling_type<T>::type beta
            = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast_nocopy<T>(mInputs[k]);
        std::shared_ptr<CudaDeviceTensor<T> > diffOutput
            = (mDiffOutputs[k].isValid())
                ? cuda_device_tensor_cast<T>(mDiffOutputs[k])
                : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[k]);

        CHECK_CUDNN_STATUS(cudnnLRNCrossChannelBackward(
            CudaContext::cudnnHandle(),
            mLRNDesc,
            CUDNN_LRN_CROSS_CHANNEL_DIM1,
            &alpha,
            mOutputDesc[k],
            mOutputs.getDevicePtr() + offset,
            mOutputDesc[k],
            mDiffInputs.getDevicePtr() + offset,
            input->getCudnnTensorDesc(),
            input->getDevicePtr(),
            &beta,
            diffOutput->getCudnnTensorDesc(),
            diffOutput->getDevicePtr()));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

template <class T>
void N2D2::LRNCell_Frame_CUDA<T>::update()
{
}

template <class T>
N2D2::LRNCell_Frame_CUDA<T>::~LRNCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mOutputDesc.size(); k < size; ++k)
        cudnnDestroyTensorDescriptor(mOutputDesc[k]);

    cudnnDestroyLRNDescriptor(mLRNDesc);
}

namespace N2D2 {
    template class LRNCell_Frame_CUDA<half_float::half>;
    template class LRNCell_Frame_CUDA<float>;
    template class LRNCell_Frame_CUDA<double>;
}

#endif
#endif
