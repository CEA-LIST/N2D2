/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#include "Cell/ConvCell_Transcode.hpp"

template <>
N2D2::Registrar<N2D2::ConvCell> N2D2::ConvCell_Transcode
    <N2D2::ConvCell_Frame<half_float::half>, N2D2::ConvCell_Spike>::mRegistrar(
        "Transcode",
        N2D2::ConvCell_Transcode
        <N2D2::ConvCell_Frame<half_float::half>, N2D2::ConvCell_Spike>::create,
        N2D2::Registrar<N2D2::ConvCell>::Type<half_float::half>());
template <>
N2D2::Registrar<N2D2::ConvCell> N2D2::ConvCell_Transcode
    <N2D2::ConvCell_Frame<float>, N2D2::ConvCell_Spike>::mRegistrar(
        "Transcode",
        N2D2::ConvCell_Transcode
        <N2D2::ConvCell_Frame<float>, N2D2::ConvCell_Spike>::create,
        N2D2::Registrar<N2D2::ConvCell>::Type<float>());
template <>
N2D2::Registrar<N2D2::ConvCell> N2D2::ConvCell_Transcode
    <N2D2::ConvCell_Frame<double>, N2D2::ConvCell_Spike>::mRegistrar(
        "Transcode",
        N2D2::ConvCell_Transcode
        <N2D2::ConvCell_Frame<double>, N2D2::ConvCell_Spike>::create,
        N2D2::Registrar<N2D2::ConvCell>::Type<double>());

#ifdef CUDA
template <>
N2D2::Registrar<N2D2::ConvCell> N2D2::ConvCell_Transcode
    <N2D2::ConvCell_Frame_CUDA<half_float::half>, N2D2::ConvCell_Spike>::mRegistrar(
        "Transcode_CUDA",
        N2D2::ConvCell_Transcode
        <N2D2::ConvCell_Frame_CUDA<half_float::half>, N2D2::ConvCell_Spike>::create,
        N2D2::Registrar<N2D2::ConvCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::ConvCell> N2D2::ConvCell_Transcode
    <N2D2::ConvCell_Frame_CUDA<float>, N2D2::ConvCell_Spike>::mRegistrar(
        "Transcode_CUDA",
        N2D2::ConvCell_Transcode
        <N2D2::ConvCell_Frame_CUDA<float>, N2D2::ConvCell_Spike>::create,
        N2D2::Registrar<N2D2::ConvCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::ConvCell> N2D2::ConvCell_Transcode
    <N2D2::ConvCell_Frame_CUDA<double>, N2D2::ConvCell_Spike>::mRegistrar(
        "Transcode_CUDA",
        N2D2::ConvCell_Transcode
        <N2D2::ConvCell_Frame_CUDA<double>, N2D2::ConvCell_Spike>::create,
        N2D2::Registrar<N2D2::ConvCell>::Type<double>());
#endif
