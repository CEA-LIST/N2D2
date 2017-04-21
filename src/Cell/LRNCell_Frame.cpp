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
    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for LRNCell " + mName);
    }
}

void N2D2::LRNCell_Frame::propagate(bool /*inference*/)
{
    if (mN > mNbOutputs)
        throw std::runtime_error("LRNCell_Frame::propagate(): mN > nbOutputs "
                                 "doesn't match for local response "
                                 "normalization accross channels\n");

    mInputs.synchronizeDToH();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (mNbOutputs > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && mNbOutputs > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < mNbOutputs; ++output) {
            int minChan = output - mN / 2;

            if (minChan < 0)
                minChan = 0;

            const unsigned int channelMin = std::max(0, minChan);
            const unsigned int channelMax
                = std::min(mNbChannels - 1, output + mN / 2);

            for (unsigned int oy = 0; oy < mInputs[0].dimY(); ++oy) {
                for (unsigned int ox = 0; ox < mInputs[0].dimX(); ++ox) {
                    // For each input channel, accumulate the value
                    Float_T accAccrossChannels = 0;

                    for (unsigned int channel = channelMin;
                         channel < channelMax;
                         ++channel)
                        accAccrossChannels += mInputs(ox, oy, output, batchPos);

                    // Compute the output signal
                    mOutputs(ox, oy, output, batchPos)
                        = normAccrossChannel(mInputs(ox, oy, output, batchPos),
                                             accAccrossChannels,
                                             mAlpha,
                                             mBeta,
                                             mK);
                }
            }
        }
    }
}

void N2D2::LRNCell_Frame::backPropagate()
{
    throw std::runtime_error(
        "LRNCell_Frame::backPropagate(): not implemented.");
}

void N2D2::LRNCell_Frame::update()
{
    if (!mDiffOutputs.empty())
        mDiffOutputs.fill(0.0);
}

float N2D2::LRNCell_Frame::normAccrossChannel(
    Float_T input, Float_T xAcc, Float_T alpha, Float_T beta, Float_T k)
{
    return input / (std::pow((k + std::pow(xAcc, 2.0) * alpha), beta));
}
