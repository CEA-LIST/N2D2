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

#ifndef N2D2_SATURATIONACTIVATION_FRAME_H
#define N2D2_SATURATIONACTIVATION_FRAME_H

#include "Activation/SaturationActivation.hpp"

namespace N2D2 {
template <class T>
class SaturationActivation_Frame : public SaturationActivation<T> {
public:
    static std::shared_ptr<SaturationActivation<T> > create()
    {
        return std::make_shared<SaturationActivation_Frame<T> >();
    }

    virtual void propagate(Tensor4d<T>* data);
    virtual void backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData);
    virtual ~SaturationActivation_Frame() {};

    using SaturationActivation<T>::mShifting;
    using SaturationActivation<T>::mThreshold;

private:
    static Registrar<SaturationActivation<T> > mRegistrar;
};
}

template <class T>
void N2D2::SaturationActivation_Frame<T>::propagate(Tensor4d<T>* data)
{
    if (mShifting > 0) {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)data->size(); ++index)
            (*data)(index) /= (1 << mShifting);
    }
    else if (mShifting < 0) {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)data->size(); ++index)
            (*data)(index) *= (1 << (-mShifting));
    }

#pragma omp parallel for if (data->size() > 1024)
    for (int index = 0; index < (int)data->size(); ++index)
        (*data)(index) = Utils::clamp<T>((*data)(index),
                                         -mThreshold, mThreshold);
}

template <class T>
void N2D2::SaturationActivation_Frame
    <T>::backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData)
{
    if (mShifting > 0) {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)data->size(); ++index)
            (*diffData)(index) /= (1 << mShifting);
    }
    else if (mShifting < 0) {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)data->size(); ++index)
            (*diffData)(index) *= (1 << (-mShifting));
    }

#pragma omp parallel for if (data->size() > 1024)
    for (int index = 0; index < (int)diffData->size(); ++index)
        (*diffData)(index)
            *= ((*data)(index) > -mThreshold && (*data)(index) < mThreshold)
                    ? 1.0f : 0.0f;
}

#endif // N2D2_SATURATIONACTIVATION_FRAME_H
