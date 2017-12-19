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

#include "Activation/LogisticActivation_Frame.hpp"

template <>
N2D2::Registrar<N2D2::LogisticActivation<N2D2::Float_T> >
N2D2::LogisticActivation_Frame
    <N2D2::Float_T>::mRegistrar(N2D2::LogisticActivation_Frame
                                <N2D2::Float_T>::create,
                                "Frame",
                                "Transcode",
                                "Spike",
                                "Spike_Analog",
                                "Spike_PCM",
                                "Spike_RRAM",
                                "CSpike",
                                NULL);

bool N2D2::LogisticActivationDisabled = false;
