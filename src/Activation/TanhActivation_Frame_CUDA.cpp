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

#include "Activation/TanhActivation_Frame_CUDA.hpp"

template <>
N2D2::Registrar<N2D2::TanhActivation<N2D2::Float_T> >
N2D2::TanhActivation_Frame_CUDA
    <N2D2::Float_T>::mRegistrar(N2D2::TanhActivation_Frame_CUDA
                                <N2D2::Float_T>::create,
                                "Frame_CUDA",
                                "Transcode_CUDA",
                                "CSpike_CUDA",
                                NULL);

template <>
void N2D2::TanhActivation_Frame_CUDA<float>::propagate(Tensor4d<float>* data)
{
    CudaTensor4d<float>* cudaData = static_cast<CudaTensor4d<float>*>(data);

    if (mAlpha != 1.0) {
        const float alpha = mAlpha;

        // data = data*mAlpha
        CHECK_CUBLAS_STATUS(cublasSscal(CudaContext::cublasHandle(),
                                        cudaData->size(),
                                        &alpha,
                                        cudaData->getDevicePtr(),
                                        1));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDNN_STATUS(cudnnActivationForward(CudaContext::cudnnHandle(),
                                              mActivationDesc,
                                              &alpha,
                                              cudaData->getCudnnTensorDesc(),
                                              cudaData->getDevicePtr(),
                                              &beta,
                                              cudaData->getCudnnTensorDesc(),
                                              cudaData->getDevicePtr()));
}

template <>
void N2D2::TanhActivation_Frame_CUDA<double>::propagate(Tensor4d<double>* data)
{
    CudaTensor4d<double>* cudaData = static_cast<CudaTensor4d<double>*>(data);

    if (mAlpha != 1.0) {
        const double alpha = mAlpha;

        // data = data*mAlpha
        CHECK_CUBLAS_STATUS(cublasDscal(CudaContext::cublasHandle(),
                                        cudaData->size(),
                                        &alpha,
                                        cudaData->getDevicePtr(),
                                        1));
    }

    const double alpha = 1.0f;
    const double beta = 0.0f;

    CHECK_CUDNN_STATUS(cudnnActivationForward(CudaContext::cudnnHandle(),
                                              mActivationDesc,
                                              &alpha,
                                              cudaData->getCudnnTensorDesc(),
                                              cudaData->getDevicePtr(),
                                              &beta,
                                              cudaData->getCudnnTensorDesc(),
                                              cudaData->getDevicePtr()));
}

#endif
