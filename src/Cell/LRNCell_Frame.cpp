/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "Cell/LRNCell_Frame.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::LRNCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::LRNCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame<float>::mRegistrar("Frame",
    N2D2::LRNCell_Frame<float>::create,
    N2D2::Registrar<N2D2::LRNCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame<double>::mRegistrar("Frame",
    N2D2::LRNCell_Frame<double>::create,
    N2D2::Registrar<N2D2::LRNCell>::Type<double>());

template <class T>
N2D2::LRNCell_Frame<T>::LRNCell_Frame(const DeepNet& deepNet, const std::string& name,
                                   unsigned int nbOutputs)
    : Cell(deepNet, name, nbOutputs),
      LRNCell(deepNet, name, nbOutputs),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
    // ctor
}

template <class T>
void N2D2::LRNCell_Frame<T>::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("LRNCell_Frame<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for LRNCell " + mName);
    }
}

template <class T>
void N2D2::LRNCell_Frame<T>::propagate(bool /*inference*/)
{
    if (mN > getNbOutputs())
        throw std::runtime_error("LRNCell_Frame<T>::propagate(): mN > nbOutputs "
                                 "doesn't match for local response "
                                 "normalization accross channels\n");

    mInputs.synchronizeDBasedToH();

    T beta(0.0f);

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const Tensor<T>& input = tensor_cast<T>(mInputs[k]);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (getNbOutputs() > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && getNbOutputs() > 16)
#endif
        for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
            for (unsigned int channel = 0; channel < input.dimZ(); ++channel) {
                const unsigned int output = channel + offset;

                const unsigned int channelMin
                    = std::max<int>(0, channel - mN / 2);
                const unsigned int channelMax
                    = std::min<size_t>(input.dimZ() - 1, channel + mN / 2);

                for (unsigned int oy = 0; oy < input.dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < input.dimX(); ++ox) {
                        // For each input channel, accumulate the value
                        T accAccrossChannels(0.0);

                        for (unsigned int accChannel = channelMin;
                            accChannel < channelMax; ++accChannel)
                        {
                            accAccrossChannels
                                += input(ox, oy, accChannel, batchPos);
                        }

                        // Compute the output signal
                        mOutputs(ox, oy, output, batchPos)
                            = normAccrossChannel(input(ox, oy, channel, batchPos),
                                                 accAccrossChannels,
                                                 T(mAlpha),
                                                 T(mBeta),
                                                 T(mK))
                            + beta * mOutputs(ox, oy, output, batchPos);
                    }
                }
            }
        }

        offset += input.dimZ();
    }

    mDiffInputs.clearValid();
}

template <class T>
void N2D2::LRNCell_Frame<T>::backPropagate()
{
    throw std::runtime_error(
        "LRNCell_Frame<T>::backPropagate(): not implemented.");
}

template <class T>
void N2D2::LRNCell_Frame<T>::update()
{
    for (unsigned int k = 0, size = mDiffOutputs.size(); k < size; ++k) {
        Tensor<T> diffOutput
            = tensor_cast_nocopy<T>(mDiffOutputs[k]);

        diffOutput.fill(T(0.0));

        mDiffOutputs[k] = diffOutput;
    }

    Cell_Frame<T>::update();
}

template <class T>
T N2D2::LRNCell_Frame<T>::normAccrossChannel(
    T input, T xAcc, T alpha, T beta, T k)
{
    const T norm(std::pow((k + (xAcc * xAcc) * alpha), beta));
    return (input / norm);
}

namespace N2D2 {
    template class LRNCell_Frame<half_float::half>;
    template class LRNCell_Frame<float>;
    template class LRNCell_Frame<double>;
}
