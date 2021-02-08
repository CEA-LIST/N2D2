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

#include "Activation/SoftplusActivation_Frame_CUDA.hpp"
#include "Cell/Cell.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::SoftplusActivation>
N2D2::SoftplusActivation_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA",
    "CSpike_LIF_CUDA"},
    N2D2::SoftplusActivation_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::SoftplusActivation>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::SoftplusActivation>
N2D2::SoftplusActivation_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA",
    "CSpike_LIF_CUDA"},
    N2D2::SoftplusActivation_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::SoftplusActivation>::Type<float>());

template <>
N2D2::Registrar<N2D2::SoftplusActivation>
N2D2::SoftplusActivation_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA",
    "CSpike_LIF_CUDA"},
    N2D2::SoftplusActivation_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::SoftplusActivation>::Type<double>());

#endif
