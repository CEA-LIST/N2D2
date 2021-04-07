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
#if CUDNN_VERSION >= 5000

#include "Cell/DropoutCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
    N2D2::DropoutCell_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::DropoutCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
    N2D2::DropoutCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::DropoutCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::DropoutCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::DropoutCell>::Type<double>());

template <class T>
N2D2::DropoutCell_Frame_CUDA<T>::DropoutCell_Frame_CUDA(const DeepNet& deepNet, 
                                                        const std::string& name,
                                                        unsigned int nbOutputs)
    : Cell(deepNet, name, nbOutputs),
      DropoutCell(deepNet, name, nbOutputs),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs),
      mStatesSize(0),
      mStates(0)
{
    int count;
    CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));
    // ctor
    mDropoutDesc.resize(count);
    for (int nb_drop=0; nb_drop<count; ++nb_drop){
        CHECK_CUDNN_STATUS(cudnnCreateDropoutDescriptor(&mDropoutDesc[nb_drop]));
    }
    //CHECK_CUDNN_STATUS(cudnnCreateDropoutDescriptor(&mDropoutDesc));

    mReserveSpaceSize.resize(count);
    mReserveSpace.resize(count);
    mOutputDesc.resize(count);
    mStates.resize(count, NULL);
    mStatesSize.resize(count);
}

template <class T>
void N2D2::DropoutCell_Frame_CUDA<T>::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("DropoutCell_Frame_CUDA<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for DropoutCell "
                                     + mName);
        }

        if (k < mOutputDesc[dev].size())
            continue;  // already initialized, skip!
            
        mOutputDesc[dev].push_back(cudnnTensorDescriptor_t());

        CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mOutputDesc[dev].back()));
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptorEx(
            mOutputDesc[dev].back(),
            CudaContext::data_type<T>::value,
            mOutputs.dimB(),
            mInputs[k].dimZ(),
            mOutputs.dimY(),
            mOutputs.dimX(),
            mOutputs.dimX() * mOutputs.dimY() * mInputs.dimZ(),
            mOutputs.dimX() * mOutputs.dimY(),
            mOutputs.dimX(),
            1));

        mReserveSpaceSize[dev].push_back(0);
        mReserveSpace[dev].push_back(NULL);

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast_nocopy<T>(mInputs[k]);

        CHECK_CUDNN_STATUS(cudnnDropoutGetReserveSpaceSize(
            input->getCudnnTensorDesc(), &mReserveSpaceSize[dev].back()));
        CHECK_CUDA_STATUS(
            cudaMalloc(&mReserveSpace[dev].back(), mReserveSpaceSize[dev].back()));
    }
    
    if (mStates[dev] == NULL) {
        CHECK_CUDNN_STATUS(
            cudnnDropoutGetStatesSize(CudaContext::cudnnHandle(), &mStatesSize[dev]));

        CHECK_CUDA_STATUS(cudaMalloc(&mStates[dev], mStatesSize[dev]));

        CHECK_CUDNN_STATUS(cudnnSetDropoutDescriptor(mDropoutDesc[dev],
                                                    CudaContext::cudnnHandle(),
                                                    mDropout,
                                                    mStates[dev],
                                                    mStatesSize[dev],
                                                    Random::mtRand()));
    }
}

template <class T>
void N2D2::DropoutCell_Frame_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));

    unsigned int offset = 0;

    if (inference) {
        if (mInputs.size() == 1) {
            std::shared_ptr<CudaDeviceTensor<T> > input0
                = cuda_device_tensor_cast<T>(mInputs[0]);

            CHECK_CUDA_STATUS(cudaMemcpy(mOutputs.getDevicePtr(),
                                         input0->getDevicePtr(),
                                         mInputs[0].size() * sizeof(T),
                                         cudaMemcpyDeviceToDevice));
        } else {
            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                std::shared_ptr<CudaDeviceTensor<T> > input
                    = cuda_device_tensor_cast<T>(mInputs[k]);

                unsigned int outputOffset = offset;
                unsigned int inputOffset = 0;

                for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
                     ++batchPos) {
                    CHECK_CUDA_STATUS(cudaMemcpy(
                        mOutputs.getDevicePtr() + outputOffset,
                        input->getDevicePtr() + inputOffset,
                        (mInputs[k].size() / mInputs.dimB()) * sizeof(T),
                        cudaMemcpyDeviceToDevice));

                    outputOffset += mOutputs.dimX() * mOutputs.dimY()
                                    * mInputs.dimZ();
                    inputOffset += mOutputs.dimX() * mOutputs.dimY()
                                   * mInputs[k].dimZ();
                }

                offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
            }
        }
    } else {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            std::shared_ptr<CudaDeviceTensor<T> > input
                = cuda_device_tensor_cast<T>(mInputs[k]);

            CHECK_CUDNN_STATUS(
                cudnnDropoutForward(CudaContext::cudnnHandle(),
                                    mDropoutDesc[dev],
                                    input->getCudnnTensorDesc(),
                                    input->getDevicePtr(),
                                    mOutputDesc[dev][k],
                                    mOutputs.getDevicePtr() + offset,
                                    mReserveSpace[dev][k],
                                    mReserveSpaceSize[dev][k]));
            offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
        }
    }

    mDiffInputs.clearValid();
}

template <class T>
void N2D2::DropoutCell_Frame_CUDA<T>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mDiffOutputs[k].isValid())
            throw std::runtime_error(
                "Cannot blend gradient from a Dropout cell");

        std::shared_ptr<CudaDeviceTensor<T> > diffOutput
            = cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[k]);

        CHECK_CUDNN_STATUS(
            cudnnDropoutBackward(CudaContext::cudnnHandle(),
                                 mDropoutDesc[dev],
                                 mOutputDesc[dev][k],
                                 mDiffInputs.getDevicePtr() + offset,
                                 diffOutput->getCudnnTensorDesc(),
                                 diffOutput->getDevicePtr(),
                                 mReserveSpace[dev][k],
                                 mReserveSpaceSize[dev][k]));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

template <class T>
void N2D2::DropoutCell_Frame_CUDA<T>::update()
{
    Cell_Frame_CUDA<T>::update();
}

template <class T>
N2D2::DropoutCell_Frame_CUDA<T>::~DropoutCell_Frame_CUDA()
{
    int dev;
    cudaGetDevice(&dev);

    for (unsigned int k = 0, size = mOutputDesc[dev].size(); k < size; ++k)
        cudnnDestroyTensorDescriptor(mOutputDesc[dev][k]);

    for (unsigned int k = 0, size = mReserveSpace[dev].size(); k < size; ++k)
        cudaFree(mReserveSpace[dev][k]);

    if (mStatesSize[dev] > 0)
        cudaFree(mStates[dev]);

    cudnnDestroyDropoutDescriptor(mDropoutDesc[dev]);
}

namespace N2D2 {
    template class DropoutCell_Frame_CUDA<half_float::half>;
    template class DropoutCell_Frame_CUDA<float>;
    template class DropoutCell_Frame_CUDA<double>;
}

#endif
#endif
