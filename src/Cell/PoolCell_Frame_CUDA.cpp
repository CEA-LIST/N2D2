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

#include "Cell/PoolCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                      N2D2::PoolCell_Frame_CUDA::create);

N2D2::PoolCell_Frame_CUDA::PoolCell_Frame_CUDA(
    const std::string& name,
    unsigned int poolWidth,
    unsigned int poolHeight,
    unsigned int nbOutputs,
    unsigned int strideX,
    unsigned int strideY,
    unsigned int paddingX,
    unsigned int paddingY,
    Pooling pooling,
    const std::shared_ptr<Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      PoolCell(name,
               poolWidth,
               poolHeight,
               nbOutputs,
               strideX,
               strideY,
               paddingX,
               paddingY,
               pooling),
      Cell_Frame_CUDA(name, nbOutputs, activation)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreatePoolingDescriptor(&mPoolingDesc));
}

void N2D2::PoolCell_Frame_CUDA::initialize()
{
    if (!isUnitMap()) {
        throw std::domain_error(
            "PoolCell_Frame_CUDA::initialize(): only unit maps are "
                "supported for cell " + mName + ".");
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for PoolCell " + mName);

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

    const cudnnPoolingMode_t poolingMode
        = (mPooling == Max) ? CUDNN_POOLING_MAX
                            : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnSetPooling2dDescriptor(
        mPoolingDesc,
        poolingMode,
        CUDNN_PROPAGATE_NAN,
        // CUDNN_NOT_PROPAGATE_NAN,
        mPoolHeight,
        mPoolWidth,
        mPaddingY,
        mPaddingX,
        mStrideY, // BUG in cuDNN v3 (order of the last 2 arguments was
        // inverted), resolved with cuDNN v5
        mStrideX));
#else
    CHECK_CUDNN_STATUS(cudnnSetPooling2dDescriptor(
        mPoolingDesc,
        poolingMode,
        mPoolHeight,
        mPoolWidth,
        mPaddingY,
        mPaddingX,
        mStrideX, // BUG in cuDNN v3 (order of the last 2 arguments was
        // inverted), resolved with cuDNN v5
        mStrideY));
#endif
}

void N2D2::PoolCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        CHECK_CUDNN_STATUS(
            cudnnPoolingForward(CudaContext::cudnnHandle(),
                                mPoolingDesc,
                                &alpha,
                                mInputs[k].getCudnnTensorDesc(),
                                mInputs[k].getDevicePtr(),
                                &beta,
                                mOutputDesc[k],
                                mOutputs.getDevicePtr() + offset));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
    }

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::PoolCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    Cell_Frame_CUDA::backPropagate();

    const float alpha = 1.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const float beta = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        CHECK_CUDNN_STATUS(
            cudnnPoolingBackward(CudaContext::cudnnHandle(),
                                 mPoolingDesc,
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

void N2D2::PoolCell_Frame_CUDA::update()
{
}

void N2D2::PoolCell_Frame_CUDA::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&PoolCell_Frame_CUDA::propagate, this, false),
                  std::bind(&PoolCell_Frame_CUDA::backPropagate, this),
                  (mPooling == Max));

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

N2D2::PoolCell_Frame_CUDA::~PoolCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mOutputDesc.size(); k < size; ++k)
        CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mOutputDesc[k]));

    CHECK_CUDNN_STATUS(cudnnDestroyPoolingDescriptor(mPoolingDesc));
}

#endif
