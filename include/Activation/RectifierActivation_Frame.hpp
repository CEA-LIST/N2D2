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

#ifndef N2D2_RECTIFIERACTIVATION_FRAME_H
#define N2D2_RECTIFIERACTIVATION_FRAME_H

#include "Activation/RectifierActivation.hpp"

namespace N2D2 {
template <class T>
class RectifierActivation_Frame : public RectifierActivation<T> {
public:
    static std::shared_ptr<RectifierActivation<T> > create()
    {
        return std::make_shared<RectifierActivation_Frame<T> >();
    }

    virtual void propagate(Tensor4d<T>* data);
    virtual void backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData);
    virtual ~RectifierActivation_Frame() {};

    using RectifierActivation<T>::mLeakSlope;
    using RectifierActivation<T>::mShifting;
    using RectifierActivation<T>::mClipping;

private:
    static Registrar<RectifierActivation<T> > mRegistrar;
};
}

template <class T>
void N2D2::RectifierActivation_Frame<T>::propagate(Tensor4d<T>* data)
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

    if (mClipping > 0.0) {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)data->size(); ++index) {
            (*data)(index) = ((*data)(index) > 0)
                ? std::min<T>((*data)(index), (T)mClipping)
                : (T)mLeakSlope * (*data)(index);
        }
    } else {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)data->size(); ++index) {
            (*data)(index) = ((*data)(index) > 0)
                ? (*data)(index)
                : (T)mLeakSlope * (*data)(index);
        }
    }
}

template <class T>
void N2D2::RectifierActivation_Frame
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

    if (mClipping > 0.0) {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)diffData->size(); ++index) {
            (*diffData)(index) *= ((*data)(index) > (T)mClipping)
                                      ? 0.0f
                                      : ((*data)(index) > 0) ? 1.0f
                                                             : (T)mLeakSlope;
        }
    } else {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)diffData->size(); ++index)
            (*diffData)(index) *= ((*data)(index) > 0) ? 1.0f : (T)mLeakSlope;
    }
}

#endif // N2D2_RECTIFIERACTIVATION_FRAME_H
