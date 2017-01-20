/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#include "Cell/PoolCell_Frame.hpp"

N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame::mRegistrar("Frame", N2D2::PoolCell_Frame::create);

N2D2::PoolCell_Frame::PoolCell_Frame(const std::string& name,
                                     unsigned int poolWidth,
                                     unsigned int poolHeight,
                                     unsigned int nbOutputs,
                                     unsigned int strideX,
                                     unsigned int strideY,
                                     unsigned int paddingX,
                                     unsigned int paddingY,
                                     Pooling pooling,
                                     const std::shared_ptr
                                     <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      PoolCell(name,
               poolWidth,
               poolHeight,
               nbOutputs,
               strideX,
               strideY,
               paddingX,
               paddingY,
               pooling),
      Cell_Frame(name, nbOutputs, activation)
{
    // ctor
}

void N2D2::PoolCell_Frame::propagate(bool inference)
{
    // Pooling preparation
    if (!inference && mPooling == Max)
        mInputsBackProp.assign(mInputs[0].dimX(),
                               mInputs[0].dimY(),
                               mInputs.dimZ(),
                               mInputs.dimB(),
                               std::vector<unsigned int>());

    if (mPooling == Max)
        mMaxLocationSwitch.assign(mInputs[0].dimX(),
                                  mInputs[0].dimY(),
                                  mInputs.dimZ(),
                                  mInputs.dimB(),
                                  (unsigned int)0);

    const unsigned int size = mInputs.dimB() * mNbOutputs;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < mNbOutputs; ++output) {
            if (mPoolNbChannels[output] == 0)
                continue; // No connection to this output...

            for (unsigned int oy = 0; oy < mOutputs.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < mOutputs.dimX(); ++ox) {
                    const unsigned int sxMax = std::min(
                        mInputs[0].dimX() - ox * mStrideX, mPoolWidth);
                    const unsigned int syMax = std::min(
                        mInputs[0].dimY() - oy * mStrideY, mPoolHeight);

                    // For each output, compute the pool value
                    Float_T poolValue = 0.0;

                    if (mPooling == Max) {
                        poolValue = -std::numeric_limits<Float_T>::infinity();
                        unsigned int channelMax = 0;
                        unsigned int ixMax = 0;
                        unsigned int iyMax = 0;

                        for (unsigned int channel = 0; channel < mNbChannels;
                             ++channel) {
                            if (!isConnection(channel, output))
                                continue;

                            for (unsigned int sy = 0; sy < syMax; ++sy) {
                                for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                    const unsigned int ix = ox * mStrideX + sx;
                                    const unsigned int iy = oy * mStrideY + sy;

                                    if (mInputs(ix, iy, channel, batchPos)
                                        > poolValue) {
                                        poolValue = mInputs(
                                            ix, iy, channel, batchPos);
                                        channelMax = channel;
                                        ixMax = ix;
                                        iyMax = iy;
                                    }
                                }
                            }
                        }

                        mMaxLocationSwitch(ixMax, iyMax, channelMax, batchPos)
                            = 1;

                        if (!inference) {
// For this output node, take the max. input across all connected channels
// Critical section needed because channelMax can be the same for several
// outputs
#pragma omp critical
                            mInputsBackProp(ixMax, iyMax, channelMax, batchPos)
                                .push_back(ox + oy * mOutputs.dimX()
                                           + output * mOutputs.dimX()
                                             * mOutputs.dimY());
                        }
                    } else if (mPooling == Average) {
                        for (unsigned int channel = 0; channel < mNbChannels;
                             ++channel) {
                            if (!isConnection(channel, output))
                                continue;

                            for (unsigned int sy = 0; sy < syMax; ++sy) {
                                for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                    const unsigned int ix = ox * mStrideX + sx;
                                    const unsigned int iy = oy * mStrideY + sy;

                                    poolValue
                                        += mInputs(ix, iy, channel, batchPos);
                                }
                            }
                        }

                        poolValue /= mPoolWidth * mPoolHeight
                                     * mPoolNbChannels[output];
                    } else {
                        assert(0 && "Unhandled pooling type!");
                    }

                    // Compute the output signal
                    mOutputs(ox, oy, output, batchPos) = poolValue;
                }
            }
        }
    }

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::PoolCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    Cell_Frame::backPropagate();

    const unsigned int oxStride
        = mStrideX * (unsigned int)((mInputs[0].dimX() - mPoolWidth + mStrideX)
                                    / (double)mStrideX);
    const unsigned int oyStride
        = mStrideY * (unsigned int)((mInputs[0].dimY() - mPoolHeight + mStrideY)
                                    / (double)mStrideY);

    const unsigned int size = mInputs.dimB() * mNbChannels;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int channel = 0; channel < mNbChannels; ++channel) {
            const bool isValid = mDiffOutputs.getTensor4d(channel).isValid();

            for (unsigned int iy = 0; iy < mInputs[0].dimY(); ++iy) {
                for (unsigned int ix = 0; ix < mInputs[0].dimX(); ++ix) {
                    Float_T gradient = 0.0;

                    if (mPooling == Max) {
                        // In max pooling, the unit which was chosen as the max
                        // receives all the error since very small changes
                        // in input would perturb the result only through that
                        // unit
                        for (std::vector<unsigned int>::const_iterator it
                             = mInputsBackProp(ix, iy, channel, batchPos)
                                   .begin(),
                             itEnd
                             = mInputsBackProp(ix, iy, channel, batchPos).end();
                             it != itEnd;
                             ++it) {
                            gradient += mDiffInputs(*it, batchPos);
                        }
                    } else if (mPooling == Average) {
                        const unsigned int sxMax = std::min(mPoolWidth, ix + 1);
                        const unsigned int syMax
                            = std::min(mPoolHeight, iy + 1);

                        for (unsigned int sy = iy % mStrideY,
                                          sx0 = ix % mStrideX;
                             sy < syMax;
                             sy += mStrideY) {
                            if (iy >= oyStride + sy)
                                continue;

                            for (unsigned int sx = sx0; sx < sxMax;
                                 sx += mStrideX) {
                                // Border conditions
                                if (ix >= oxStride + sx)
                                    continue;

                                // Output node coordinates
                                const unsigned int ox = (ix - sx) / mStrideX;
                                const unsigned int oy = (iy - sy) / mStrideY;

                                for (unsigned int output = 0;
                                     output < mNbOutputs;
                                     ++output) {
                                    if (!isConnection(channel, output))
                                        continue;

                                    gradient
                                        += mDiffInputs(ox, oy, output, batchPos)
                                           / mPoolNbChannels[output];
                                }
                            }
                        }

                        // In mean pooling, uniformly distribute the error for a
                        // single pooling unit among the units which feed
                        // into it in the previous layer
                        gradient /= (mPoolHeight * mPoolWidth);
                    } else {
                        assert(0 && "Unhandled pooling type!");
                    }

                    mDiffOutputs(ix, iy, channel, batchPos)
                        = gradient + isValid
                                     * mDiffOutputs(ix, iy, channel, batchPos);
                }
            }
        }
    }

    mDiffOutputs.setValid();
}

void N2D2::PoolCell_Frame::update()
{
}

void N2D2::PoolCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&PoolCell_Frame::propagate, this, false),
                  std::bind(&PoolCell_Frame::backPropagate, this));

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 0; in < mInputs.size(); ++in) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << in << "]";

            gc.check(name.str(),
                     mInputs[in],
                     mDiffOutputs[in],
                     (mPooling == Max) ? &mInputsBackProp : NULL);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}
