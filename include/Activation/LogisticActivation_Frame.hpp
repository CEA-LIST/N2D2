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

#ifndef WIN32
#include <fenv.h>
#endif

#include "Activation/LogisticActivation.hpp"
#include "Cell/Cell.hpp"
#include "containers/Tensor.hpp"
#include "Solver/SGDSolver_Kernels.hpp"

namespace N2D2 {
template <class T>
class LogisticActivation_Frame : public LogisticActivation {
public:
    static std::shared_ptr<LogisticActivation> create(bool withLoss = false)
    {
        return std::make_shared<LogisticActivation_Frame<T> >(withLoss);
    }

    LogisticActivation_Frame(bool withLoss = false);
    virtual void propagate(const Cell& cell, BaseTensor& data, bool inference = false);
    virtual void backPropagate(const Cell& cell, BaseTensor& data, BaseTensor& diffData);
    virtual ~LogisticActivation_Frame() {};

private:
    static Registrar<LogisticActivation> mRegistrar;
};
}

template <class T>
N2D2::LogisticActivation_Frame<T>::LogisticActivation_Frame(bool withLoss)
    : LogisticActivation(withLoss)
{
    // ctor
}

template <class T>
void N2D2::LogisticActivation_Frame<T>::propagate(const Cell& cell, 
                                                  BaseTensor& baseData, bool inference)
{
    if (LogisticActivationDisabled)
        return;

    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

    mScaling.propagate(cell, data);

#pragma omp parallel for if (data.size() > 1024)
    for (int index = 0; index < (int)data.size(); ++index){
#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
        const int excepts = fegetexcept();
        fedisableexcept(FE_OVERFLOW);
#endif

        data(index) = 1.0f / (1.0f + std::exp(-data(index)));

#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
        feenableexcept(excepts);
#endif
    }

    if (mQuantizationLevels > 0) {
        ++mNbSteps;

        if (mNbSteps > mQuantizationDelay || inference) {
            quantize(data, data, T(0.0f), T(1.0f),
                     (unsigned int)mQuantizationLevels);
        }
    }
}

template <class T>
void N2D2::LogisticActivation_Frame<T>::backPropagate(const Cell& cell, 
                                                      BaseTensor& baseData, BaseTensor& baseDiffData)
{
    if (LogisticActivationDisabled)
        return;

    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);
    Tensor<T>& diffData = dynamic_cast<Tensor<T>&>(baseDiffData);
    

    if (mQuantizationLevels > 0) {
#pragma omp parallel for if (diffData.size() > 1024)
        for (int index = 0; index < (int)diffData.size(); ++index) {
            diffData(index) = Utils::clamp<T>(diffData(index),
                                              T(-1.0f), T(1.0f));
        }
    }

    if (!this->mWithLoss) {
#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)diffData.size(); ++index)
            diffData(index) *= data(index) * (1.0f - data(index));
    }
    
    mScaling.backPropagate(cell, data, diffData);
}

#endif // N2D2_LOGISTICACTIVATION_FRAME_H
