/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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
#include "Cell/SoftmaxCell_Frame.hpp"
#include "DeepNet.hpp"

template <>
N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::SoftmaxCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::SoftmaxCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame<float>::mRegistrar("Frame",
    N2D2::SoftmaxCell_Frame<float>::create,
    N2D2::Registrar<N2D2::SoftmaxCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame<double>::mRegistrar("Frame",
    N2D2::SoftmaxCell_Frame<double>::create,
    N2D2::Registrar<N2D2::SoftmaxCell>::Type<double>());

template <class T>
N2D2::SoftmaxCell_Frame<T>::SoftmaxCell_Frame(const DeepNet& deepNet, const std::string& name,
                                           unsigned int nbOutputs,
                                           bool withLoss,
                                           unsigned int groupSize)
    : Cell(deepNet, name, nbOutputs),
      SoftmaxCell(deepNet, name, nbOutputs, withLoss, groupSize),
      Cell_Frame<T>(deepNet, name, nbOutputs)
{
    // ctor
}

template <class T>
void N2D2::SoftmaxCell_Frame<T>::initialize()
{
    if (mInputs.size() > 1)
        throw std::domain_error("SoftmaxCell_Frame<T>::initialize(): inputs "
                                "concatenation is not supported.");
/*
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("SoftmaxCell_Frame<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }
*/

    if(mGroupSize > 0)
    {
        if(getNbOutputs() % mGroupSize)
            throw std::domain_error("SoftmaxCell_Frame<T>::initialize():"
                                    " the group size must be divisible by "
                                    "the number of outputs.");

    }
}

template <class T>
void N2D2::SoftmaxCell_Frame<T>::propagate(bool /*inference*/)
{
    mInputs.synchronizeDBasedToH();
    const unsigned int groupStride = mGroupSize > 0 ? mGroupSize : getNbOutputs();

    const Tensor<T>& input = tensor_cast<T>(mInputs[0]);

#pragma omp parallel for if (mInputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int oy = 0; oy < mOutputsDims[1]; ++oy) {
            for (unsigned int ox = 0; ox < mOutputsDims[0]; ++ox) {
                for(unsigned int step = 0; step < getNbOutputs()/groupStride; ++step)
                {
                    const unsigned int stride = step*mGroupSize;
                    const unsigned int nbNeurons = stride + groupStride;

                    T maxVal(input(ox, oy, stride, batchPos));

                    for (unsigned int output = stride + 1; output < nbNeurons; ++output)
                        maxVal
                            = std::max(maxVal, input(ox, oy, output, batchPos));

                    // double required for large number of channels
                    T sum(0.0);

                    for (unsigned int output = stride; output < nbNeurons; ++output)
                        sum += std::exp(input(ox, oy, output, batchPos) - maxVal);

                    if (sum > T(0.0)) {
                        for (unsigned int output = stride; output < nbNeurons; ++output)
                            mOutputs(ox, oy, output, batchPos)
                                = std::exp(input(ox, oy, output, batchPos)
                                        - maxVal) / sum;
                    } else {
                        for (unsigned int output = stride; output < nbNeurons; ++output)
                            mOutputs(ox, oy, output, batchPos) = T(0.0);
                    }
                }
            }
        }
    }

    mDiffInputs.clearValid();
}

template <class T>
void N2D2::SoftmaxCell_Frame<T>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    const unsigned int size = mInputs.dimB() * getNbChannels();
    const unsigned int groupStride = mGroupSize > 0 ? mGroupSize : getNbOutputs();

    const T beta((mDiffOutputs[0].isValid()) ? 1.0 : 0.0);

    Tensor<T> diffOutput = (mDiffOutputs[0].isValid())
        ? tensor_cast<T>(mDiffOutputs[0])
        : tensor_cast_nocopy<T>(mDiffOutputs[0]);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
        {
            for (unsigned int iy = 0; iy < mInputs[0].dimY(); ++iy) {
                for (unsigned int ix = 0; ix < mInputs[0].dimX(); ++ix) {
                    if (mWithLoss) {
                        diffOutput(ix, iy, channel, batchPos)
                            = mDiffInputs(ix, iy, channel, batchPos)
                              + beta
                                * diffOutput(ix, iy, channel, batchPos);
                    } else {
                        T gradient(0.0);

                        for(unsigned int step = 0; step < getNbOutputs()/groupStride; ++step)
                        {
                            const unsigned int stride = step*mGroupSize;
                            const unsigned int nbNeurons = stride + groupStride;

                            for (unsigned int output = stride; output < nbNeurons;
                                ++output) {
                                gradient += ((output == channel)
                                            - mOutputs(ix, iy, channel, batchPos))
                                            * mOutputs(ix, iy, output, batchPos)
                                            * mDiffInputs(ix, iy, output, batchPos);
                            }
                        }
                        diffOutput(ix, iy, channel, batchPos)
                            = gradient
                              + beta
                                * diffOutput(ix, iy, channel, batchPos);
                    }
                }
            }
        }
    }

    mDiffOutputs[0] = diffOutput;
    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

template <class T>
void N2D2::SoftmaxCell_Frame<T>::update()
{
}

template <class T>
void N2D2::SoftmaxCell_Frame<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&SoftmaxCell_Frame<T>::propagate, this, false),
                  std::bind(&SoftmaxCell_Frame<T>::backPropagate, this));

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 0; in < mInputs.size(); ++in) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << in << "]";

            gc.check(name.str(), mInputs[in], mDiffOutputs[in]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

namespace N2D2 {
    template class SoftmaxCell_Frame<half_float::half>;
    template class SoftmaxCell_Frame<float>;
    template class SoftmaxCell_Frame<double>;
}
