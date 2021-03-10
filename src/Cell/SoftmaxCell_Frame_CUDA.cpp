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

#ifdef CUDA

#include "Cell/SoftmaxCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"
#include "CublasUtils.hpp"

template <>
N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
    N2D2::SoftmaxCell_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::SoftmaxCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
    N2D2::SoftmaxCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::SoftmaxCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::SoftmaxCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::SoftmaxCell>::Type<double>());

template <class T>
N2D2::SoftmaxCell_Frame_CUDA<T>::SoftmaxCell_Frame_CUDA(const DeepNet& deepNet, 
                                                     const std::string& name,
                                                     unsigned int nbOutputs,
                                                     bool withLoss,
                                                     unsigned int groupSize)
    : Cell(deepNet, name, nbOutputs),
      SoftmaxCell(deepNet, name, nbOutputs, withLoss, groupSize),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs)
{
    // ctor
    if (mGroupSize > 0)
        CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mGroupTensor));
}

template <class T>
void N2D2::SoftmaxCell_Frame_CUDA<T>::initialize()
{
    if (mInputs.size() > 1)
        throw std::domain_error("SoftmaxCell_Frame_CUDA<T>::initialize(): inputs "
                                "concatenation is not supported.");

    if(mGroupSize > 0)
    {
        if(getNbOutputs() % mGroupSize)
            throw std::domain_error("SoftmaxCell_Frame::initialize():"
                                    " the group size must be divisible by "
                                    "the number of outputs.");

        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptorEx(
            mGroupTensor,
            CudaContext::data_type<T>::value,
            mInputs[0].dimB(),
            mGroupSize,
            mInputs[0].dimY(),
            mInputs[0].dimX(),
            mInputs[0].dimZ()*mInputs[0].dimY()*mInputs[0].dimX(),
            mInputs[0].dimY()*mInputs[0].dimX(),
            mInputs[0].dimX(),
            1));

    }
}

template <class T>
void N2D2::SoftmaxCell_Frame_CUDA<T>::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;
    const typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

    std::shared_ptr<CudaDeviceTensor<T> > input0
        = cuda_device_tensor_cast_nocopy<T>(mInputs[0]);

    if(mGroupSize > 0)
    {
        for(unsigned int step = 0; step < getNbOutputs(); step += mGroupSize)
        {
            const unsigned int offset = step*mInputs[0].dimX()*mInputs[0].dimY();

            CHECK_CUDNN_STATUS(cudnnSoftmaxForward(CudaContext::cudnnHandle(),
                                                CUDNN_SOFTMAX_ACCURATE,
                                                CUDNN_SOFTMAX_MODE_CHANNEL,
                                                &alpha,
                                                mGroupTensor,
                                                input0->getDevicePtr() + offset,
                                                &beta,
                                                mGroupTensor,
                                                mOutputs.getDevicePtr() + offset));
        }

    }
    else
    {

        CHECK_CUDNN_STATUS(cudnnSoftmaxForward(CudaContext::cudnnHandle(),
                                            CUDNN_SOFTMAX_ACCURATE,
                                            CUDNN_SOFTMAX_MODE_CHANNEL,
                                            &alpha,
                                            input0->getCudnnTensorDesc(),
                                            input0->getDevicePtr(),
                                            &beta,
                                            mOutputs.getCudnnTensorDesc(),
                                            mOutputs.getDevicePtr()));
    }
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::SoftmaxCell_Frame_CUDA<T>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    if (mWithLoss)
        backPropagateWithLoss();
    else {
        const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;
        const typename Cuda::cudnn_scaling_type<T>::type beta
            = (mDiffOutputs[0].isValid()) ? 1.0f : 0.0f;

        std::shared_ptr<CudaDeviceTensor<T> > diffOutput0
            = (mDiffOutputs[0].isValid())
                ? cuda_device_tensor_cast<T>(mDiffOutputs[0])
                : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[0]);

        if(mGroupSize > 0) {
            for(unsigned int step = 0; step < getNbOutputs(); step += mGroupSize)
            {
                const unsigned int offset = step*mInputs[0].dimX()*mInputs[0].dimY();
                CHECK_CUDNN_STATUS(
                    cudnnSoftmaxBackward(CudaContext::cudnnHandle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &alpha,
                                        mGroupTensor,
                                        mOutputs.getDevicePtr() + offset,
                                        mGroupTensor,
                                        mDiffInputs.getDevicePtr() + offset,
                                        &beta,
                                        mGroupTensor,
                                        diffOutput0->getDevicePtr() + offset));
            }

        }
        else {
            CHECK_CUDNN_STATUS(
                cudnnSoftmaxBackward(CudaContext::cudnnHandle(),
                                     CUDNN_SOFTMAX_ACCURATE,
                                     CUDNN_SOFTMAX_MODE_CHANNEL,
                                     &alpha,
                                     mOutputs.getCudnnTensorDesc(),
                                     mOutputs.getDevicePtr(),
                                     mDiffInputs.getCudnnTensorDesc(),
                                     mDiffInputs.getDevicePtr(),
                                     &beta,
                                     diffOutput0->getCudnnTensorDesc(),
                                     diffOutput0->getDevicePtr()));
        }

        mDiffOutputs[0].deviceTensor() = *diffOutput0;
    }

    mDiffOutputs[0].setValid();
    mDiffOutputs.synchronizeDToHBased();

}

template <class T>
void N2D2::SoftmaxCell_Frame_CUDA<T>::backPropagateWithLoss() {
    const T alpha(1.0f);

    std::shared_ptr<CudaDeviceTensor<T> > diffOutput0
        = (mDiffOutputs[0].isValid())
            ? cuda_device_tensor_cast<T>(mDiffOutputs[0])
            : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[0]);

    if (mDiffOutputs[0].isValid()) {

        CHECK_CUBLAS_STATUS(
            cublasAxpy(CudaContext::cublasHandle(),
                        mDiffOutputs[0].size(), // size of data
                        &alpha,
                        mDiffInputs.getDevicePtr(),
                        1,
                        diffOutput0->getDevicePtr(),
                        1));
    } else {
    
        CHECK_CUDA_STATUS(
            cudaMemcpy(diffOutput0->getDevicePtr(),
                       mDiffInputs.getDevicePtr(),
                       mDiffOutputs[0].size() * sizeof(T),
                       cudaMemcpyDeviceToDevice));

      
    }
    /*
                if (mInputs.dimB() > 1) {
                    float normBatch = 1.0f/mInputs.dimB();

                    //Normalized in function of the batch size
                    CHECK_CUBLAS_STATUS(
       cublasSscal(CudaContext::cublasHandle(),
                        mDiffOutputs[0].size(),
                        &normBatch,
                        mDiffOutputs[0].getDevicePtr(),
                        1) );
                }
    */
    mDiffOutputs[0].deviceTensor() = *diffOutput0;
}

template <class T>
void N2D2::SoftmaxCell_Frame_CUDA<T>::update()
{
    Cell_Frame_CUDA<T>::update();
}

template <class T>
N2D2::SoftmaxCell_Frame_CUDA<T>::~SoftmaxCell_Frame_CUDA()
{
    if(mGroupSize > 0)
        cudnnDestroyTensorDescriptor(mGroupTensor);

}

namespace N2D2 {
    template class SoftmaxCell_Frame_CUDA<half_float::half>;
    template class SoftmaxCell_Frame_CUDA<float>;
    template class SoftmaxCell_Frame_CUDA<double>;
}

#endif
