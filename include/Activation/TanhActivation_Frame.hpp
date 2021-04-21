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

#ifndef N2D2_TANHACTIVATION_FRAME_H
#define N2D2_TANHACTIVATION_FRAME_H

#include "Activation/TanhActivation.hpp"
#include "Cell/Cell.hpp"
#include "containers/Tensor.hpp"
#include "Solver/SGDSolver_Kernels.hpp"

namespace N2D2 {
template <class T> class TanhActivation_Frame : public TanhActivation {
public:
    static std::shared_ptr<TanhActivation> create()
    {
        return std::make_shared<TanhActivation_Frame<T> >();
    }

    virtual void propagate(const Cell& cell,
                           const BaseTensor& input,
                           BaseTensor& output,
                           bool inference = false);
    virtual void backPropagate(const Cell& cell,
                               const BaseTensor& input,
                               const BaseTensor& output,
                               const BaseTensor& diffInput,
                               BaseTensor& diffOutput);
    virtual void update(unsigned int batchSize);
    virtual ~TanhActivation_Frame() {};

private:
    static Registrar<TanhActivation> mRegistrar;
};
}

template <class T>
void N2D2::TanhActivation_Frame<T>::propagate(
    const Cell& cell, 
    const BaseTensor& baseInput,
    BaseTensor& baseOutput,
    bool inference)
{
    const Tensor<T>& input = dynamic_cast<const Tensor<T>&>(baseInput);
    Tensor<T>& output = dynamic_cast<Tensor<T>&>(baseOutput);
    //If activations is quantized : use Q Level of activations for saturate    
    //Else : Use Q Level of weights parameters 
    const std::size_t nbbits = mQuantizedNbBits > 0 ? 
                                mQuantizedNbBits : cell.getQuantizedNbBits();
    mScaling.propagate(cell, input, output, nbbits);

    if (mAlpha != 1.0) {
        const T alpha(mAlpha);

#pragma omp parallel for if (output.size() > 1024)
        for (int index = 0; index < (int)output.size(); ++index)
            output(index) = std::tanh(alpha * output(index));
    } else {
#pragma omp parallel for if (output.size() > 1024)
        for (int index = 0; index < (int)output.size(); ++index)
            output(index) = std::tanh(output(index));
    }
    if(mQuantizer) {
        mQuantizer->propagate(baseOutput, inference);
    }
}

template <class T>
void N2D2::TanhActivation_Frame<T>::backPropagate(
    const Cell& cell, 
    const BaseTensor& /*baseInput*/,
    const BaseTensor& baseOutput,
    const BaseTensor& baseDiffInput,
    BaseTensor& baseDiffOutput)
{
    if(mQuantizer) {
        mQuantizer->back_propagate( mQuantizer->getFullPrecisionActivations(), 
                                    baseOutput,/*Not use for the moment*/
                                    baseDiffInput,
                                    baseDiffOutput);
    }
    const Tensor<T>& output = dynamic_cast<const Tensor<T>&>(baseOutput);
    const Tensor<T>& diffInput = (!mQuantizer)  ? dynamic_cast<const Tensor<T>&>(baseDiffInput) 
                                : dynamic_cast<const Tensor<T>&>(baseDiffOutput);
    Tensor<T>& diffOutput = dynamic_cast<Tensor<T>&>(baseDiffOutput);

    if (mAlpha != 1.0) {
        const T alpha(mAlpha);

#pragma omp parallel for if (output.size() > 1024)
        for (int index = 0; index < (int)diffOutput.size(); ++index)
            diffOutput(index) = diffInput(index) * alpha * (1.0f - output(index) * output(index));
    } else {
#pragma omp parallel for if (output.size() > 1024)
        for (int index = 0; index < (int)diffOutput.size(); ++index)
            diffOutput(index) = diffInput(index) * (1.0f - output(index) * output(index));
    }
    
    mScaling.backPropagate(cell, diffOutput, diffOutput);
}

template <class T>
void N2D2::TanhActivation_Frame<T>::update(unsigned int batchSize)
{
    if(mQuantizer) {
        mQuantizer->update(batchSize);
    }
}
#endif // N2D2_TANHACTIVATION_FRAME_H
