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

#ifndef N2D2_LOGISTICACTIVATION_FRAME_H
#define N2D2_LOGISTICACTIVATION_FRAME_H

#include "Activation/LogisticActivation.hpp"

namespace N2D2 {
template <class T>
class LogisticActivation_Frame : public LogisticActivation<T> {
public:
    static std::shared_ptr<LogisticActivation<T> > create(bool withLoss = false)
    {
        return std::make_shared<LogisticActivation_Frame<T> >(withLoss);
    }

    LogisticActivation_Frame(bool withLoss = false);
    virtual void propagate(Tensor4d<T>* data);
    virtual void backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData);
    virtual ~LogisticActivation_Frame() {};

private:
    static Registrar<LogisticActivation<T> > mRegistrar;
};
}

template <class T>
N2D2::LogisticActivation_Frame<T>::LogisticActivation_Frame(bool withLoss)
    : LogisticActivation<T>(withLoss)
{
    // ctor
}

template <class T>
void N2D2::LogisticActivation_Frame<T>::propagate(Tensor4d<T>* data)
{
    if (LogisticActivationDisabled)
        return;

#pragma omp parallel for if (data->size() > 1024)
    for (int index = 0; index < (int)data->size(); ++index){
#if !defined(WIN32) && !defined(__APPLE__)
        const int excepts = fegetexcept();
        fedisableexcept(FE_OVERFLOW);
#endif

        (*data)(index) = 1.0f / (1.0f + std::exp(-(*data)(index)));

#if !defined(WIN32) && !defined(__APPLE__)
        feenableexcept(excepts);
#endif
    }

}

template <class T>
void N2D2::LogisticActivation_Frame
    <T>::backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData)
{
    if (LogisticActivationDisabled)
        return;

    if (!this->mWithLoss) {
#pragma omp parallel for if (data->size() > 1024)
        for (int index = 0; index < (int)diffData->size(); ++index)
            (*diffData)(index) *= (*data)(index) * (1.0f - (*data)(index));
    }
}

#endif // N2D2_LOGISTICACTIVATION_FRAME_H
