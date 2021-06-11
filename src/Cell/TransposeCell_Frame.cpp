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
#include "Cell/TransposeCell_Frame.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::TransposeCell>
N2D2::TransposeCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::TransposeCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::TransposeCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::TransposeCell>
N2D2::TransposeCell_Frame<float>::mRegistrar("Frame",
    N2D2::TransposeCell_Frame<float>::create,
    N2D2::Registrar<N2D2::TransposeCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::TransposeCell>
N2D2::TransposeCell_Frame<double>::mRegistrar("Frame",
    N2D2::TransposeCell_Frame<double>::create,
    N2D2::Registrar<N2D2::TransposeCell>::Type<double>());

template <class T>
N2D2::TransposeCell_Frame<T>::TransposeCell_Frame(const DeepNet& deepNet, const std::string& name,
                                 unsigned int nbOutputs,
                                 const std::vector<int>& perm)
    : Cell(deepNet, name, nbOutputs),
      TransposeCell(deepNet, name, nbOutputs, perm),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
    // ctor
}

template <class T>
void N2D2::TransposeCell_Frame<T>::initialize()
{
    if (mInputs.size() > 1) {
        throw std::domain_error("TransposeCell_Frame<T>::initialize(): "
                                "inputs concatenation is not supported.");
    }
}


template <class T>
void N2D2::TransposeCell_Frame<T>::initializeDataDependent()
{
    Cell_Frame<T>::initializeDataDependent();
    initialize();
}

template <class T>
void N2D2::TransposeCell_Frame<T>::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    const Tensor<T>& input = tensor_cast<T>(mInputs[0]);

    std::size_t coords[4];
    for (coords[3] = 0; coords[3] < input.dims()[3]; ++coords[3]) {
        for (coords[2] = 0; coords[2] < input.dims()[2]; ++coords[2]) {
            for (coords[1] = 0; coords[1] < input.dims()[1]; ++coords[1]) {
                for (coords[0] = 0; coords[0] < input.dims()[0]; ++coords[0]) {
                    mOutputs(coords[mPerm[0]], coords[mPerm[1]],
                             coords[mPerm[2]], coords[mPerm[3]])
                        = input(coords[0], coords[1], coords[2], coords[3]);
                }
            }
        }
    }

    Cell_Frame<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::TransposeCell_Frame<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame<T>::backPropagate();

    if (!mDiffOutputs.empty()) {
        const std::vector<int> invPerm = getInversePermutation();

        Tensor<T> diffOutputs = tensor_cast<T>(mDiffOutputs[0]);

        std::size_t coords[4];
        for (coords[3] = 0; coords[3] < mDiffInputs.dims()[3]; ++coords[3]) {
            for (coords[2] = 0; coords[2] < mDiffInputs.dims()[2]; ++coords[2]) {
                for (coords[1] = 0; coords[1] < mDiffInputs.dims()[1];
                    ++coords[1])
                {
                    for (coords[0] = 0; coords[0] < mDiffInputs.dims()[0];
                        ++coords[0])
                    {
                        diffOutputs(coords[invPerm[0]], coords[invPerm[1]],
                                    coords[invPerm[2]], coords[invPerm[3]])
                            = mDiffInputs(coords[0], coords[1],
                                          coords[2], coords[3]);
                    }
                }
            }
        }

        mDiffOutputs[0] = diffOutputs;

        mDiffOutputs[0].setValid();
        mDiffOutputs[0].synchronizeHToD();
    }
}

template <class T>
void N2D2::TransposeCell_Frame<T>::update()
{
    
}

template <class T>
void N2D2::TransposeCell_Frame<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&TransposeCell_Frame<T>::propagate, this, false),
                  std::bind(&TransposeCell_Frame<T>::backPropagate, this));

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
N2D2::TransposeCell_Frame<T>::~TransposeCell_Frame()
{
    //dtor
}

namespace N2D2 {
    template class TransposeCell_Frame<half_float::half>;
    template class TransposeCell_Frame<float>;
    template class TransposeCell_Frame<double>;
}
