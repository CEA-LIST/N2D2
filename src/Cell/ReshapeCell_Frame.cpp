/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "GradientCheck.hpp"
#include "Cell/ReshapeCell_Frame.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::ReshapeCell>
N2D2::ReshapeCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::ReshapeCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::ReshapeCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::ReshapeCell>
N2D2::ReshapeCell_Frame<float>::mRegistrar("Frame",
    N2D2::ReshapeCell_Frame<float>::create,
    N2D2::Registrar<N2D2::ReshapeCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::ReshapeCell>
N2D2::ReshapeCell_Frame<double>::mRegistrar("Frame",
    N2D2::ReshapeCell_Frame<double>::create,
    N2D2::Registrar<N2D2::ReshapeCell>::Type<double>());

template <class T>
N2D2::ReshapeCell_Frame<T>::ReshapeCell_Frame(const DeepNet& deepNet, const std::string& name,
                                 unsigned int nbOutputs,
                                 const std::vector<int>& dims)
    : Cell(deepNet, name, nbOutputs),
      ReshapeCell(deepNet, name, nbOutputs, dims),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
    // ctor
}

template <class T>
void N2D2::ReshapeCell_Frame<T>::initialize()
{
    if (mInputs.size() > 1) {
        throw std::domain_error("ReshapeCell_Frame<T>::initialize(): "
                                "inputs concatenation is not supported.");
    }
}



template <class T>
void N2D2::ReshapeCell_Frame<T>::initializeDataDependent()
{
    Cell_Frame<T>::initializeDataDependent();
    initialize();
}


template <class T>
void N2D2::ReshapeCell_Frame<T>::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    const Tensor<T>& input = tensor_cast<T>(mInputs[0]);
    // TODO: replace copy with DataTensor pointer copy
    std::copy(input.begin(), input.end(), mOutputs.begin());

    Cell_Frame<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::ReshapeCell_Frame<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame<T>::backPropagate();

    if (!mDiffOutputs.empty()) {
        Tensor<T> diffOutputs = tensor_cast<T>(mDiffOutputs[0]);
        // TODO: replace copy with DataTensor pointer copy
        std::copy(mDiffInputs.begin(), mDiffInputs.end(), diffOutputs.begin());
        mDiffOutputs[0] = diffOutputs;

        mDiffOutputs[0].setValid();
        mDiffOutputs[0].synchronizeHToD();
    }
}

template <class T>
void N2D2::ReshapeCell_Frame<T>::update()
{
    
}

template <class T>
void N2D2::ReshapeCell_Frame<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ReshapeCell_Frame<T>::propagate, this, false),
                  std::bind(&ReshapeCell_Frame<T>::backPropagate, this));

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0; k < mInputs.size(); ++k) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

template <class T>
N2D2::ReshapeCell_Frame<T>::~ReshapeCell_Frame()
{
    //dtor
}

namespace N2D2 {
    template class ReshapeCell_Frame<half_float::half>;
    template class ReshapeCell_Frame<float>;
    template class ReshapeCell_Frame<double>;
}
