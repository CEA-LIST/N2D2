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

N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                         N2D2::DropoutCell_Frame_CUDA::create);

N2D2::DropoutCell_Frame_CUDA::DropoutCell_Frame_CUDA(const std::string& name,
                                                     unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      DropoutCell(name, nbOutputs),
      Cell_Frame_CUDA(name, nbOutputs),
      mStatesSize(0),
      mStates(NULL)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateDropoutDescriptor(&mDropoutDesc));
}

void N2D2::DropoutCell_Frame_CUDA::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("DropoutCell_Frame_CUDA::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    CHECK_CUDNN_STATUS(
        cudnnDropoutGetStatesSize(CudaContext::cudnnHandle(), &mStatesSize));
    CHECK_CUDA_STATUS(cudaMalloc(&mStates, mStatesSize));

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for DropoutCell "
                                     + mName);
        }

        mOutputDesc.push_back(cudnnTensorDescriptor_t());

        CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mOutputDesc.back()));
        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptorEx(
            mOutputDesc.back(),
            CudaContext::data_type,
            mOutputs.dimB(),
            mInputs[k].dimZ(),
            mOutputs.dimY(),
            mOutputs.dimX(),
            mOutputs.dimX() * mOutputs.dimY() * mInputs.dimZ(),
            mOutputs.dimX() * mOutputs.dimY(),
            mOutputs.dimX(),
            1));

        mReserveSpaceSize.push_back(0);
        mReserveSpace.push_back(NULL);

        CHECK_CUDNN_STATUS(cudnnDropoutGetReserveSpaceSize(
            mInputs[k].getCudnnTensorDesc(), &mReserveSpaceSize.back()));
        CHECK_CUDA_STATUS(
            cudaMalloc(&mReserveSpace.back(), mReserveSpaceSize.back()));
    }

    CHECK_CUDNN_STATUS(cudnnSetDropoutDescriptor(mDropoutDesc,
                                                 CudaContext::cudnnHandle(),
                                                 mDropout,
                                                 mStates,
                                                 mStatesSize,
                                                 Random::mtRand()));
}

void N2D2::DropoutCell_Frame_CUDA::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    unsigned int offset = 0;

    if (inference) {
        if (mInputs.size() == 1) {
            CHECK_CUDA_STATUS(cudaMemcpy(mOutputs.getDevicePtr(),
                                         mInputs[0].getDevicePtr(),
                                         mInputs[0].size() * sizeof(Float_T),
                                         cudaMemcpyDeviceToDevice));
        } else {
            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                unsigned int outputOffset = offset;
                unsigned int inputOffset = 0;

                for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
                     ++batchPos) {
                    CHECK_CUDA_STATUS(cudaMemcpy(
                        mOutputs.getDevicePtr() + outputOffset,
                        mInputs[k].getDevicePtr() + inputOffset,
                        (mInputs[k].size() / mInputs.dimB()) * sizeof(Float_T),
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
            CHECK_CUDNN_STATUS(
                cudnnDropoutForward(CudaContext::cudnnHandle(),
                                    mDropoutDesc,
                                    mInputs[k].getCudnnTensorDesc(),
                                    mInputs[k].getDevicePtr(),
                                    mOutputDesc[k],
                                    mOutputs.getDevicePtr() + offset,
                                    mReserveSpace[k],
                                    mReserveSpaceSize[k]));

            offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
        }
    }

    mDiffInputs.clearValid();
}

void N2D2::DropoutCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mDiffOutputs[k].isValid())
            throw std::runtime_error(
                "Cannot blend gradient from a Dropout cell");

        CHECK_CUDNN_STATUS(
            cudnnDropoutBackward(CudaContext::cudnnHandle(),
                                 mDropoutDesc,
                                 mOutputDesc[k],
                                 mDiffInputs.getDevicePtr() + offset,
                                 mDiffOutputs[k].getCudnnTensorDesc(),
                                 mDiffOutputs[k].getDevicePtr(),
                                 mReserveSpace[k],
                                 mReserveSpaceSize[k]));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::DropoutCell_Frame_CUDA::update()
{
}

N2D2::DropoutCell_Frame_CUDA::~DropoutCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mOutputDesc.size(); k < size; ++k)
        CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mOutputDesc[k]));

    for (unsigned int k = 0, size = mReserveSpace.size(); k < size; ++k)
        CHECK_CUDA_STATUS(cudaFree(mReserveSpace[k]));

    if (mStatesSize > 0)
        CHECK_CUDA_STATUS(cudaFree(mStates));

    CHECK_CUDNN_STATUS(cudnnDestroyDropoutDescriptor(mDropoutDesc));
}

#endif
#endif
