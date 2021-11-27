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
#include "GradientCheck.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame.hpp"
#include "Cell/NormalizeCell.hpp"
#include "Cell/NormalizeCell_Frame.hpp"
#include "containers/Tensor.hpp"


static const N2D2::Registrar<N2D2::NormalizeCell> registrarHalfFloat(
                    "Frame", N2D2::NormalizeCell_Frame<half_float::half>::create,
                    N2D2::Registrar<N2D2::NormalizeCell>::Type<half_float::half>());

static const N2D2::Registrar<N2D2::NormalizeCell> registrarFloat(
                    "Frame", N2D2::NormalizeCell_Frame<float>::create,
                    N2D2::Registrar<N2D2::NormalizeCell>::Type<float>());

static const N2D2::Registrar<N2D2::NormalizeCell> registrarDouble(
                    "Frame", N2D2::NormalizeCell_Frame<double>::create,
                    N2D2::Registrar<N2D2::NormalizeCell>::Type<double>());


template<class T>
N2D2::NormalizeCell_Frame<T>::NormalizeCell_Frame(const DeepNet& deepNet, const std::string& name,
                                              unsigned int nbOutputs, Norm norm)
    : Cell(deepNet, name, nbOutputs),
      NormalizeCell(deepNet, name, nbOutputs, std::move(norm)),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
}

template<class T>
void N2D2::NormalizeCell_Frame<T>::initialize() {
    if(mInputs.size() != 1) {
        throw std::runtime_error("There can only be one input for NormalizeCell '" + mName + "'.");
    }

    if(mInputs[0].size() != mOutputs.size()) {
        throw std::runtime_error("The size of the input and output of cell '" + mName + "' must be the same");
    }

    mNormData.resize(mOutputs.dims());
}

template<class T>
void N2D2::NormalizeCell_Frame<T>::propagate(bool /*inference*/) {
    mInputs.synchronizeDBasedToH();

    const Tensor<T>& input = tensor_cast<T>(mInputs[0]);

    if (mNorm == L2) {
#pragma omp parallel for if (mInputs.dimB() > 4)
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int oy = 0; oy < mOutputsDims[1]; ++oy) {
                for (unsigned int ox = 0; ox < mOutputsDims[0]; ++ox) {
                    T sumSq(0.0);

                    for (unsigned int output = 0; output < mOutputs.dimZ();
                        ++output)
                    {
                        sumSq += input(ox, oy, output, batchPos)
                                    * input(ox, oy, output, batchPos);
                    }

                    const T scale(std::sqrt(sumSq + 1.0e-6));

                    for (unsigned int output = 0; output < mOutputs.dimZ();
                        ++output)
                    {
                        mNormData(ox, oy, output, batchPos) = scale;
                        mOutputs(ox, oy, output, batchPos)
                            = input(ox, oy, output, batchPos) / scale;
                    }
                }
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported norm.");
    }

    Cell_Frame<T>::propagate();
    mDiffInputs.clearValid();
}

template<class T>
void N2D2::NormalizeCell_Frame<T>::backPropagate() {
    if (mDiffOutputs[0].empty() || !mDiffInputs.isValid())
        return;

    Cell_Frame<T>::backPropagate();

    const T beta((mDiffOutputs[0].isValid()) ? 1.0 : 0.0);

    Tensor<T> diffOutput = (mDiffOutputs[0].isValid())
        ? tensor_cast<T>(mDiffOutputs[0])
        : tensor_cast_nocopy<T>(mDiffOutputs[0]);

    if (mNorm == L2) {
#pragma omp parallel for if (mInputs.dimB() > 4)
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int oy = 0; oy < mOutputsDims[1]; ++oy) {
                for (unsigned int ox = 0; ox < mOutputsDims[0]; ++ox) {
                    T a(0.0);

                    for (unsigned int output = 0; output < mOutputs.dimZ();
                        ++output)
                    {
                        a += mOutputs(ox, oy, output, batchPos)
                            * mDiffInputs(ox, oy, output, batchPos);
                    }

                    for (unsigned int output = 0; output < mOutputs.dimZ();
                        ++output)
                    {
                        diffOutput(ox, oy, output, batchPos)
                            = (mDiffInputs(ox, oy, output, batchPos)
                                - mOutputs(ox, oy, output, batchPos) * a)
                                    / mNormData(ox, oy, output, batchPos)
                              + beta * diffOutput(ox, oy, output, batchPos);
                    }
                }
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported norm.");
    }

    mDiffOutputs[0] = diffOutput;
    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

template<class T>
void N2D2::NormalizeCell_Frame<T>::update() {

    Cell_Frame<T>::update();
}

template<class T>
void N2D2::NormalizeCell_Frame<T>::checkGradient(double epsilon, double maxError) {
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&NormalizeCell_Frame<T>::propagate, this, false),
                  std::bind(&NormalizeCell_Frame<T>::backPropagate, this));

    for (unsigned int k = 0; k < mInputs.size(); ++k) {
        if (mDiffOutputs[k].empty()) {
            std::cout << Utils::cwarning << "Empty diff. outputs #" << k
                    << " for cell " << mName
                    << ", could not check the gradient!" << Utils::cdef
                    << std::endl;
            continue;
        }

        std::stringstream name;
        name << mName + "_mDiffOutputs[" << k << "]";

        gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
    }
}

namespace N2D2 {
    template class NormalizeCell_Frame<half_float::half>;
    template class NormalizeCell_Frame<float>;
    template class NormalizeCell_Frame<double>;
}
