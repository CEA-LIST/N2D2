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

N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                         N2D2::SoftmaxCell_Frame_CUDA::create);

N2D2::SoftmaxCell_Frame_CUDA::SoftmaxCell_Frame_CUDA(const std::string& name,
                                                     unsigned int nbOutputs,
                                                     bool withLoss)
    : Cell(name, nbOutputs),
      SoftmaxCell(name, nbOutputs, withLoss),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
}

void N2D2::SoftmaxCell_Frame_CUDA::initialize()
{
    if (mInputs.size() > 1)
        throw std::domain_error("SoftmaxCell_Frame_CUDA::initialize(): inputs "
                                "concatenation is not supported.");
}

void N2D2::SoftmaxCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDNN_STATUS(cudnnSoftmaxForward(CudaContext::cudnnHandle(),
                                           CUDNN_SOFTMAX_ACCURATE,
                                           CUDNN_SOFTMAX_MODE_CHANNEL,
                                           &alpha,
                                           mInputs[0].getCudnnTensorDesc(),
                                           mInputs[0].getDevicePtr(),
                                           &beta,
                                           mOutputs.getCudnnTensorDesc(),
                                           mOutputs.getDevicePtr()));

    mDiffInputs.clearValid();
}

void N2D2::SoftmaxCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    const float alpha = 1.0f;

    if (mWithLoss) {
        if (mDiffOutputs[0].isValid()) {
            CHECK_CUBLAS_STATUS(
                cublasSaxpy(CudaContext::cublasHandle(),
                            mDiffOutputs[0].size(), // size of data
                            &alpha,
                            mDiffInputs.getDevicePtr(),
                            1,
                            mDiffOutputs[0].getDevicePtr(),
                            1));
        } else {
            CHECK_CUDA_STATUS(
                cudaMemcpy(mDiffOutputs[0].getDevicePtr(),
                           mDiffInputs.getDevicePtr(),
                           mDiffOutputs[0].size() * sizeof(Float_T),
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
    } else {
        const float beta = (mDiffOutputs[0].isValid()) ? 1.0f : 0.0f;

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
                                 mDiffOutputs[0].getCudnnTensorDesc(),
                                 mDiffOutputs[0].getDevicePtr()));
    }

    mDiffOutputs[0].setValid();
    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::SoftmaxCell_Frame_CUDA::update()
{
}

N2D2::SoftmaxCell_Frame_CUDA::~SoftmaxCell_Frame_CUDA()
{
}

#endif
