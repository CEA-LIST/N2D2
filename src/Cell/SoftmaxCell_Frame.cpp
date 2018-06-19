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

#include "Cell/SoftmaxCell_Frame.hpp"

N2D2::Registrar<N2D2::SoftmaxCell>
N2D2::SoftmaxCell_Frame::mRegistrar("Frame", N2D2::SoftmaxCell_Frame::create);

N2D2::SoftmaxCell_Frame::SoftmaxCell_Frame(const std::string& name,
                                           unsigned int nbOutputs,
                                           bool withLoss,
                                           unsigned int groupSize)
    : Cell(name, nbOutputs),
      SoftmaxCell(name, nbOutputs, withLoss, groupSize),
      Cell_Frame(name, nbOutputs)
{
    // ctor
}

void N2D2::SoftmaxCell_Frame::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("SoftmaxCell_Frame::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }
    if(mGroupSize > 0)
    {
        if(getNbOutputs() % mGroupSize)
            throw std::domain_error("SoftmaxCell_Frame::initialize():"
                                    " the group size must be divisible by "
                                    "the number of outputs.");

    }
}

void N2D2::SoftmaxCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDToH();
    const unsigned int groupStride = mGroupSize > 0 ? mGroupSize : getNbOutputs();

#pragma omp parallel for if (mInputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int oy = 0; oy < mOutputsDims[1]; ++oy) {
            for (unsigned int ox = 0; ox < mOutputsDims[0]; ++ox) {
                for(unsigned int step = 0; step < getNbOutputs()/groupStride; ++step)
                {
                    const unsigned int stride = step*mGroupSize;
                    const unsigned int nbNeurons = stride + groupStride;

                    Float_T maxVal = mInputs(ox, oy, stride, batchPos);

                    for (unsigned int output = stride + 1; output < nbNeurons; ++output)
                        maxVal
                            = std::max(maxVal, mInputs(ox, oy, output, batchPos));

                    // double required for large number of channels
                    double sum = 0.0;

                    for (unsigned int output = stride; output < nbNeurons; ++output)
                        sum += std::exp(mInputs(ox, oy, output, batchPos) - maxVal);

                    if (sum > 0.0) {
                        for (unsigned int output = stride; output < nbNeurons; ++output)
                            mOutputs(ox, oy, output, batchPos)
                                = std::exp(mInputs(ox, oy, output, batchPos)
                                        - maxVal) / sum;
                    } else {
                        for (unsigned int output = stride; output < nbNeurons; ++output)
                            mOutputs(ox, oy, output, batchPos) = 0.0;
                    }
                }
            }
        }
    }

    mDiffInputs.clearValid();
}

void N2D2::SoftmaxCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    const unsigned int size = mInputs.dimB() * getNbChannels();
    const unsigned int groupStride = mGroupSize > 0 ? mGroupSize : getNbOutputs();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
        {
            const bool isValid = mDiffOutputs.getTensor(channel).isValid();

            for (unsigned int iy = 0; iy < mInputs[0].dimY(); ++iy) {
                for (unsigned int ix = 0; ix < mInputs[0].dimX(); ++ix) {
                    if (mWithLoss) {
                        mDiffOutputs(ix, iy, channel, batchPos)
                            = mDiffInputs(ix, iy, channel, batchPos)
                              + isValid
                                * mDiffOutputs(ix, iy, channel, batchPos);
                    } else {
                        Float_T gradient = 0.0;

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
                        mDiffOutputs(ix, iy, channel, batchPos)
                            = gradient
                              + isValid
                                * mDiffOutputs(ix, iy, channel, batchPos);
                    }
                }
            }
        }
    }

    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

void N2D2::SoftmaxCell_Frame::update()
{
}

void N2D2::SoftmaxCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&SoftmaxCell_Frame::propagate, this, false),
                  std::bind(&SoftmaxCell_Frame::backPropagate, this));

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
