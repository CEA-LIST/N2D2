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
#ifdef CUDA

#include <stdexcept>
#include <string>

#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_CUDA.hpp"
#include "Cell/ScalingCell.hpp"
#include "Cell/ScalingCell_Frame_CUDA.hpp"
#include "containers/Tensor.hpp"
#include "DeepNet.hpp"
#include "utils/Utils.hpp"


static const N2D2::Registrar<N2D2::ScalingCell> registrarHalfFloat(
                    "Frame_CUDA", N2D2::ScalingCell_Frame_CUDA<half_float::half>::create,
                    N2D2::Registrar<N2D2::ScalingCell>::Type<half_float::half>());

static const N2D2::Registrar<N2D2::ScalingCell> registrarFloat(
                    "Frame_CUDA", N2D2::ScalingCell_Frame_CUDA<float>::create,
                    N2D2::Registrar<N2D2::ScalingCell>::Type<float>());

static const N2D2::Registrar<N2D2::ScalingCell> registrarDouble(
                    "Frame_CUDA", N2D2::ScalingCell_Frame_CUDA<double>::create,
                    N2D2::Registrar<N2D2::ScalingCell>::Type<double>());


template<class T>
N2D2::ScalingCell_Frame_CUDA<T>::ScalingCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                                        unsigned int nbOutputs, Scaling scaling)
    : Cell(deepNet, name, nbOutputs),
      ScalingCell(deepNet, name, nbOutputs, scaling),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs)
{
}

template<class T>
void N2D2::ScalingCell_Frame_CUDA<T>::initialize() {
    if(mInputs.size() != 1) {
        throw std::runtime_error("There can only be one input for ScalingCell '" + mName + "'.");
    }

    if(mInputs[0].size() != mOutputs.size()) {
        throw std::runtime_error("The size of the input and output of cell '" + mName + "' must be the same");
    }
}

template<class T>
void N2D2::ScalingCell_Frame_CUDA<T>::propagate(bool inference) {
    mInputs.synchronizeHBasedToD();
    
    const CudaTensor<T>& input = cuda_tensor_cast<T>(mInputs[0]);
    mScaling.propagate(*this, input, mOutputs);
    
    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template<class T>
void N2D2::ScalingCell_Frame_CUDA<T>::backPropagate() {
    throw std::runtime_error("backPropagate not supported yet.");
}


template<class T>
void N2D2::ScalingCell_Frame_CUDA<T>::update() {
    // Nothing to update
}

template<class T>
void N2D2::ScalingCell_Frame_CUDA<T>::checkGradient(double /*epsilon*/, double /*maxError*/) {
    throw std::runtime_error("checkGradient not supported yet.");
}

template<class T>
std::pair<double, double> N2D2::ScalingCell_Frame_CUDA<T>::getOutputsRange() const {
    const auto& activation = Cell_Frame_CUDA<T>::getActivation();
    return activation?activation->getOutputRange():ScalingCell::getOutputsRange();
}

#endif