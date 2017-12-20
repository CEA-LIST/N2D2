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

#include "Activation/SaturationActivation_Frame_CUDA.hpp"

template <>
N2D2::Registrar<N2D2::SaturationActivation<N2D2::Float_T> >
N2D2::SaturationActivation_Frame_CUDA
    <N2D2::Float_T>::mRegistrar(N2D2::SaturationActivation_Frame_CUDA
                                <N2D2::Float_T>::create,
                                "Frame_CUDA",
                                "Transcode_CUDA",
                                "CSpike_CUDA",
                                NULL);

namespace N2D2 {
template <>
void SaturationActivation_Frame_CUDA<float>::propagate(Tensor4d<float>* data)
{
    CudaTensor4d<float>* cudaData = static_cast<CudaTensor4d<float>*>(data);
    cudaSSaturation_propagate(cudaData->getDevicePtr(),
                              cudaData->size(),
                              (int)mShifting,
                              (double)mThreshold);
}

template <>
void SaturationActivation_Frame_CUDA<double>::propagate(Tensor4d<double>* data)
{
    CudaTensor4d<double>* cudaData = static_cast<CudaTensor4d<double>*>(data);
    cudaDSaturation_propagate(cudaData->getDevicePtr(),
                              cudaData->size(),
                              (int)mShifting,
                              (double)mThreshold);
}

template <>
void SaturationActivation_Frame_CUDA
    <float>::backPropagate(Tensor4d<float>* data, Tensor4d<float>* diffData)
{
    CudaTensor4d<float>* cudaData = static_cast<CudaTensor4d<float>*>(data);
    CudaTensor4d<float>* cudaDiffData = static_cast
        <CudaTensor4d<float>*>(diffData);
    cudaSSaturation_backPropagate(cudaData->getDevicePtr(),
                                  cudaDiffData->getDevicePtr(),
                                  cudaData->size(),
                                  (int)mShifting,
                                  (double)mThreshold);
}

template <>
void SaturationActivation_Frame_CUDA
    <double>::backPropagate(Tensor4d<double>* data, Tensor4d<double>* diffData)
{
    CudaTensor4d<double>* cudaData = static_cast<CudaTensor4d<double>*>(data);
    CudaTensor4d<double>* cudaDiffData = static_cast
        <CudaTensor4d<double>*>(diffData);
    cudaDSaturation_backPropagate(cudaData->getDevicePtr(),
                                  cudaDiffData->getDevicePtr(),
                                  cudaData->size(),
                                  (int)mShifting,
                                  (double)mThreshold);
}
}

#endif
