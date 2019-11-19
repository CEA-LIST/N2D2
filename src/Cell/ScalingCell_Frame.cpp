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
#include "Scaling.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/ScalingCell_Frame.hpp"
#include "containers/Tensor.hpp"


static const N2D2::Registrar<N2D2::ScalingCell> registrarHalfFloat(
                    "Frame", N2D2::ScalingCell_Frame<half_float::half>::create,
                    N2D2::Registrar<N2D2::ScalingCell>::Type<half_float::half>());

static const N2D2::Registrar<N2D2::ScalingCell> registrarFloat(
                    "Frame", N2D2::ScalingCell_Frame<float>::create,
                    N2D2::Registrar<N2D2::ScalingCell>::Type<float>());

static const N2D2::Registrar<N2D2::ScalingCell> registrarDouble(
                    "Frame", N2D2::ScalingCell_Frame<double>::create,
                    N2D2::Registrar<N2D2::ScalingCell>::Type<double>());


template<class T>
N2D2::ScalingCell_Frame<T>::ScalingCell_Frame(const DeepNet& deepNet, const std::string& name,
                                              unsigned int nbOutputs, Scaling scaling)
    : Cell(deepNet, name, nbOutputs),
      ScalingCell(deepNet, name, nbOutputs, std::move(scaling)),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
}

template<class T>
void N2D2::ScalingCell_Frame<T>::initialize() {
    if(mInputs.size() != 1) {
        throw std::runtime_error("There can only be one input for ScalingCell '" + mName + "'.");
    }

    if(mInputs[0].size() != mOutputs.size()) {
        throw std::runtime_error("The size of the input and output of cell '" + mName + "' must be the same");
    }
}

template<class T>
void N2D2::ScalingCell_Frame<T>::propagate(bool /*inference*/) {
    mInputs.synchronizeDBasedToH();

    const Tensor<T>& input = tensor_cast<T>(mInputs[0]);
    mScaling.propagate(input, mOutputs);

    Cell_Frame<T>::propagate();
    mDiffInputs.clearValid();
}

template<class T>
void N2D2::ScalingCell_Frame<T>::backPropagate() {
    throw std::runtime_error("backPropagate not supported yet.");
}

template<class T>
void N2D2::ScalingCell_Frame<T>::update() {
    // Nothing to update
}

template<class T>
void N2D2::ScalingCell_Frame<T>::checkGradient(double /*epsilon*/, double /*maxError*/) {
    throw std::runtime_error("checkGradient not supported yet.");
}

template<class T>
std::pair<double, double> N2D2::ScalingCell_Frame<T>::getOutputsRange() const {
    const auto& activation = Cell_Frame<T>::getActivation();
    return activation?activation->getOutputRange():ScalingCell::getOutputsRange();
}
