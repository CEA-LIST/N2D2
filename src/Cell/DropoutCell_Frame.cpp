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

#include "Cell/DropoutCell_Frame.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::DropoutCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::DropoutCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame<float>::mRegistrar("Frame",
    N2D2::DropoutCell_Frame<float>::create,
    N2D2::Registrar<N2D2::DropoutCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame<double>::mRegistrar("Frame",
    N2D2::DropoutCell_Frame<double>::create,
    N2D2::Registrar<N2D2::DropoutCell>::Type<double>());

template <class T>
N2D2::DropoutCell_Frame<T>::DropoutCell_Frame(const DeepNet& deepNet, const std::string& name,
                                              unsigned int nbOutputs)
    : Cell(deepNet, name, nbOutputs),
      DropoutCell(deepNet, name, nbOutputs),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
    // ctor
}

template <class T>
void N2D2::DropoutCell_Frame<T>::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("DropoutCell_Frame<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for DropoutCell "
                                     + mName);
        }
    }

    mMask.resize(mOutputs.dims());
}

template <class T>
void N2D2::DropoutCell_Frame<T>::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    unsigned int offset = 0;

    if (inference) {
        if (mInputs.size() == 1)
            mOutputs = mInputs[0];
        else {
            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                const Tensor<T>& input = tensor_cast<T>(mInputs[k]);

                unsigned int outputOffset = offset;
                unsigned int inputOffset = 0;

                for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
                    ++batchPos)
                {
                    std::copy(input.begin() + inputOffset,
                              input.begin() + inputOffset
                                + (mInputs[k].size() / mInputs.dimB()),
                              mOutputs.begin() + outputOffset);

                    outputOffset += mOutputs.dimX() * mOutputs.dimY()
                                    * mInputs.dimZ();
                    inputOffset += mOutputs.dimX() * mOutputs.dimY()
                                   * mInputs[k].dimZ();
                }

                offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
            }
        }
    } else {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const Tensor<T>& input = tensor_cast<T>(mInputs[k]);

            unsigned int outputOffset = offset;
            unsigned int inputOffset = 0;

            for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
                ++batchPos)
            {
                for (unsigned int index = 0,
                     batchSize = (mInputs[k].size() / mInputs.dimB());
                     index < batchSize;
                     ++index)
                {
                    const unsigned int outputIndex = index + outputOffset;

                    mMask(outputIndex) = Random::randBernoulli(1.0 - mDropout);
                    mOutputs(outputIndex) = (mMask(outputIndex))
                        ? input(index + inputOffset)
                        : 0.0;
                }

                outputOffset += mOutputs.dimX() * mOutputs.dimY()
                                * mInputs.dimZ();
                inputOffset += mOutputs.dimX() * mOutputs.dimY()
                               * mInputs[k].dimZ();
            }

            offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
        }
    }

    mDiffInputs.clearValid();
}

template <class T>
void N2D2::DropoutCell_Frame<T>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mDiffOutputs[k].isValid())
            throw std::runtime_error(
                "Cannot blend gradient from a Dropout cell");

        Tensor<T> diffOutput
            = tensor_cast_nocopy<T>(mDiffOutputs[k]);

        unsigned int outputOffset = offset;
        unsigned int inputOffset = 0;

        for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
            ++batchPos)
        {
            for (unsigned int index = 0,
                 batchSize = (mInputs[k].size() / mInputs.dimB());
                 index < batchSize;
                 ++index)
            {
                const unsigned int outputIndex = index + outputOffset;
                diffOutput(index + inputOffset) = (mMask(outputIndex))
                    ? mDiffInputs(outputIndex)
                    : 0.0;
            }

            outputOffset += mOutputs.dimX() * mOutputs.dimY()
                            * mInputs.dimZ();
            inputOffset += mOutputs.dimX() * mOutputs.dimY()
                           * mInputs[k].dimZ();
        }

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();

        mDiffOutputs[k] = diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

template <class T>
void N2D2::DropoutCell_Frame<T>::update()
{
    Cell_Frame<T>::update();
}

template <class T>
N2D2::DropoutCell_Frame<T>::~DropoutCell_Frame()
{
    //dtor
}

namespace N2D2 {
    template class DropoutCell_Frame<half_float::half>;
    template class DropoutCell_Frame<float>;
    template class DropoutCell_Frame<double>;
}
