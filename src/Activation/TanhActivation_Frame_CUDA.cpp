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
N2D2::Registrar<N2D2::TanhActivation>
N2D2::TanhActivation_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA"},
    N2D2::TanhActivation_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::TanhActivation>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::TanhActivation>
N2D2::TanhActivation_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA"},
    N2D2::TanhActivation_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::TanhActivation>::Type<float>());

template <>
N2D2::Registrar<N2D2::TanhActivation>
N2D2::TanhActivation_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA"},
    N2D2::TanhActivation_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::TanhActivation>::Type<double>());

namespace N2D2 {
template <>
void TanhActivation_Frame_CUDA<half_float::half>::propagate(
    CudaTensor<half_float::half>& data,
    bool inference)
{
    if (mQuantizationLevels > 0) {
        ++mNbSteps;

        if (mNbSteps > mQuantizationDelay || inference) {
            cudaHquantize(data.getDevicePtr(),
                          data.getDevicePtr(),
                          data.size(),
                          half_float::half(-1.0f),
                          half_float::half(1.0f),
                          mQuantizationLevels);

        }
    }
}

template <>
void TanhActivation_Frame_CUDA<float>::propagate(CudaTensor<float>& data,
                                                 bool inference)
{
    if (mQuantizationLevels > 0) {
        ++mNbSteps;

        if (mNbSteps > mQuantizationDelay || inference) {
            cudaSquantize(data.getDevicePtr(),
                          data.getDevicePtr(),
                          data.size(),
                          -1.0f,
                          1.0f,
                          mQuantizationLevels);

        }
    }
}

template <>
void TanhActivation_Frame_CUDA<double>::propagate(CudaTensor<double>& data,
                                                  bool inference)
{
    if (mQuantizationLevels > 0) {
        ++mNbSteps;

        if (mNbSteps > mQuantizationDelay || inference) {
            cudaDquantize(data.getDevicePtr(),
                          data.getDevicePtr(),
                          data.size(),
                          -1.0,
                          1.0,
                          mQuantizationLevels);

        }
    }
}
}

#endif
