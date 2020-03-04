/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <stdexcept>
#include <string>

#include "DeepNet.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame.hpp"
#include "Cell/TargetBiasCell.hpp"
#include "Cell/TargetBiasCell_Frame.hpp"
#include "containers/Tensor.hpp"


static const N2D2::Registrar<N2D2::TargetBiasCell> registrarHalfFloat(
                    "Frame", N2D2::TargetBiasCell_Frame<half_float::half>::create,
                    N2D2::Registrar<N2D2::TargetBiasCell>::Type<half_float::half>());

static const N2D2::Registrar<N2D2::TargetBiasCell> registrarFloat(
                    "Frame", N2D2::TargetBiasCell_Frame<float>::create,
                    N2D2::Registrar<N2D2::TargetBiasCell>::Type<float>());

static const N2D2::Registrar<N2D2::TargetBiasCell> registrarDouble(
                    "Frame", N2D2::TargetBiasCell_Frame<double>::create,
                    N2D2::Registrar<N2D2::TargetBiasCell>::Type<double>());


template<class T>
N2D2::TargetBiasCell_Frame<T>::TargetBiasCell_Frame(const DeepNet& deepNet, const std::string& name,
                                              unsigned int nbOutputs, double bias)
    : Cell(deepNet, name, nbOutputs),
      TargetBiasCell(deepNet, name, nbOutputs, std::move(bias)),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
}

template<class T>
void N2D2::TargetBiasCell_Frame<T>::initialize() {
    if(mInputs.size() != 1) {
        throw std::runtime_error("There can only be one input for TargetBiasCell '" + mName + "'.");
    }

    if(mInputs[0].size() != mOutputs.size()) {
        throw std::runtime_error("The size of the input and output of cell '" + mName + "' must be the same");
    }
}

template<class T>
void N2D2::TargetBiasCell_Frame<T>::propagate(bool inference) {
    mInputs.synchronizeDBasedToH();

    const Tensor<T>& input = tensor_cast<T>(mInputs[0]);

    if (!inference) {
        for (unsigned int index = 0, size = mOutputs.size(); index < size;
            ++index)
        {
            mOutputs(index) = input(index);

            if (mDiffInputs(index) > 0.0 && input(index) > -mBias)
                mOutputs(index) += mBias;
        }
    }
    else
        mOutputs = input;

    Cell_Frame<T>::propagate();
    mDiffInputs.clearValid();
}

template<class T>
void N2D2::TargetBiasCell_Frame<T>::backPropagate() {
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame<T>::backPropagate();

    const T beta((mDiffOutputs[0].isValid()) ? 1.0 : 0.0);

    Tensor<T> diffOutput = (mDiffOutputs[0].isValid())
        ? tensor_cast<T>(mDiffOutputs[0])
        : tensor_cast_nocopy<T>(mDiffOutputs[0]);

    for (unsigned int index = 0, size = mDiffInputs.size(); index < size;
        ++index)
    {
        diffOutput(index) = mDiffInputs(index) + beta * diffOutput(index);
    }

    mDiffOutputs[0] = diffOutput;
    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

template<class T>
void N2D2::TargetBiasCell_Frame<T>::update() {
    // Nothing to update
}

template<class T>
void N2D2::TargetBiasCell_Frame<T>::checkGradient(double /*epsilon*/, double /*maxError*/) {
    throw std::runtime_error("checkGradient not supported yet.");
}

namespace N2D2 {
    template class TargetBiasCell_Frame<half_float::half>;
    template class TargetBiasCell_Frame<float>;
    template class TargetBiasCell_Frame<double>;
}
