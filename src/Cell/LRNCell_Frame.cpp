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

N2D2::Registrar<N2D2::LRNCell>
N2D2::LRNCell_Frame::mRegistrar("Frame", N2D2::LRNCell_Frame::create);

N2D2::LRNCell_Frame::LRNCell_Frame(const std::string& name,
                                   unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      LRNCell(name, nbOutputs),
      Cell_Frame(name, nbOutputs)
{
    // ctor
}

void N2D2::LRNCell_Frame::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("LRNCell_Frame::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for LRNCell " + mName);
    }
}

void N2D2::LRNCell_Frame::propagate(bool /*inference*/)
{
    if (mN > getNbOutputs())
        throw std::runtime_error("LRNCell_Frame::propagate(): mN > nbOutputs "
                                 "doesn't match for local response "
                                 "normalization accross channels\n");

    mInputs.synchronizeDToH();

    float beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[k]);

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
                        Float_T accAccrossChannels = 0;

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
                                                 mAlpha,
                                                 mBeta,
                                                 mK)
                            + beta * mOutputs(ox, oy, output, batchPos);
                    }
                }
            }
        }

        offset += input.dimZ();
    }

    mDiffInputs.clearValid();
}

void N2D2::LRNCell_Frame::backPropagate()
{
    throw std::runtime_error(
        "LRNCell_Frame::backPropagate(): not implemented.");
}

void N2D2::LRNCell_Frame::update()
{
    for (unsigned int k = 0, size = mDiffOutputs.size(); k < size; ++k) {
        Tensor<Float_T> diffOutput
            = tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

        diffOutput.fill(0.0);

        mDiffOutputs[k] = diffOutput;
    }
}

float N2D2::LRNCell_Frame::normAccrossChannel(
    Float_T input, Float_T xAcc, Float_T alpha, Float_T beta, Float_T k)
{
    return input / (std::pow((k + std::pow(xAcc, 2.0) * alpha), beta));
}
