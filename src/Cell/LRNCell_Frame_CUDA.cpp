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

N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                     N2D2::LRNCell_Frame_CUDA::create);

N2D2::LRNCell_Frame_CUDA::LRNCell_Frame_CUDA(const std::string& name,
                                             unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      LRNCell(name, nbOutputs),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateLRNDescriptor(&mLRNDesc));
}

void N2D2::LRNCell_Frame_CUDA::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("LRNCell_Frame_CUDA::initialize():"
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
            CudaContext::data_type,
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

void N2D2::LRNCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        CHECK_CUDNN_STATUS(
            cudnnLRNCrossChannelForward(CudaContext::cudnnHandle(),
                                        mLRNDesc,
                                        CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                        &alpha,
                                        mInputs[k].getCudnnTensorDesc(),
                                        mInputs[k].getDevicePtr(),
                                        &beta,
                                        mOutputDesc[k],
                                        mOutputs.getDevicePtr() + offset));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
    }

    mDiffInputs.clearValid();
}

void N2D2::LRNCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    const float alpha = 1.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const float beta = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        CHECK_CUDNN_STATUS(cudnnLRNCrossChannelBackward(
            CudaContext::cudnnHandle(),
            mLRNDesc,
            CUDNN_LRN_CROSS_CHANNEL_DIM1,
            &alpha,
            mOutputDesc[k],
            mOutputs.getDevicePtr() + offset,
            mOutputDesc[k],
            mDiffInputs.getDevicePtr() + offset,
            mInputs[k].getCudnnTensorDesc(),
            mInputs[k].getDevicePtr(),
            &beta,
            mDiffOutputs[k].getCudnnTensorDesc(),
            mDiffOutputs[k].getDevicePtr()));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::LRNCell_Frame_CUDA::update()
{
}

N2D2::LRNCell_Frame_CUDA::~LRNCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mOutputDesc.size(); k < size; ++k)
        CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mOutputDesc[k]));

    CHECK_CUDNN_STATUS(cudnnDestroyLRNDescriptor(mLRNDesc));
}

#endif
#endif
