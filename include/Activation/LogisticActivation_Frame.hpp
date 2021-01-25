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
    virtual void propagate(const Cell& cell,
                           const BaseTensor& input,
                           BaseTensor& output,
                           bool inference = false);
    virtual void backPropagate(const Cell& cell,
                               const BaseTensor& input,
                               const BaseTensor& output,
                               const BaseTensor& diffInput,
                               BaseTensor& diffOutput);
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
void N2D2::LogisticActivation_Frame<T>::propagate(
    const Cell& cell, 
    const BaseTensor& baseInput,
    BaseTensor& baseOutput,
    bool inference)
{
    if (LogisticActivationDisabled)
        return;

    const Tensor<T>& input = dynamic_cast<const Tensor<T>&>(baseInput);
    Tensor<T>& output = dynamic_cast<Tensor<T>&>(baseOutput);

    mScaling.propagate(cell, input, output);

#pragma omp parallel for if (output.size() > 1024)
    for (int index = 0; index < (int)output.size(); ++index){
#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
        const int excepts = fegetexcept();
        fedisableexcept(FE_OVERFLOW);
#endif

        output(index) = 1.0f / (1.0f + std::exp(-output(index)));

#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
        feenableexcept(excepts);
#endif
    }
    if(mQuantizer) {
        mQuantizer->propagate(baseOutput, inference);
    }
}

template <class T>
void N2D2::LogisticActivation_Frame<T>::backPropagate(
    const Cell& cell, 
    const BaseTensor& /*baseInput*/,
    const BaseTensor& baseOutput,
    const BaseTensor& baseDiffInput,
    BaseTensor& baseDiffOutput)
{
    if (LogisticActivationDisabled)
        return;

    const Tensor<T>& output = dynamic_cast<const Tensor<T>&>(baseOutput);
    const Tensor<T>& diffInput = dynamic_cast<const Tensor<T>&>(baseDiffInput);
    Tensor<T>& diffOutput = dynamic_cast<Tensor<T>&>(baseDiffOutput);

    if (!this->mWithLoss) {
#pragma omp parallel for if (output.size() > 1024)
        for (int index = 0; index < (int)diffOutput.size(); ++index)
            diffOutput(index) = diffInput(index)
                * (output(index) * (1.0f - output(index)));

        mScaling.backPropagate(cell, diffOutput, diffOutput);
    }
    else
        mScaling.backPropagate(cell, diffInput, diffOutput);
}

#endif // N2D2_LOGISTICACTIVATION_FRAME_H
