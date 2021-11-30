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

#include "GradientCheck.hpp"
#include "Cell/FMPCell_Frame.hpp"
#include "DeepNet.hpp"
#include "StimuliProvider.hpp"

N2D2::Registrar<N2D2::FMPCell>
N2D2::FMPCell_Frame::mRegistrar("Frame", N2D2::FMPCell_Frame::create);

N2D2::FMPCell_Frame::FMPCell_Frame(const DeepNet& deepNet, const std::string& name,
                                   double scalingRatio,
                                   unsigned int nbOutputs,
                                   const std::shared_ptr
                                   <Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      FMPCell(deepNet, name, scalingRatio, nbOutputs),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs, activation),
      mLockRandom(false)
{
    // ctor
}

void N2D2::FMPCell_Frame::initialize()
{
    FMPCell::initialize();

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for FMPCell " + mName);
    }

    // Generate initial regions for checkGradient()
    generateRegions(mGridX, mInputs[0].dimX(), mOutputs.dimX());
    generateRegions(mGridY, mInputs[0].dimY(), mOutputs.dimY());

    logRegions(mName + "_region.log");
}

void N2D2::FMPCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    if (!inference)
        mInputsBackProp.assign({mInputs[0].dimX(),
                               mInputs[0].dimY(),
                               mInputs.dimZ(),
                               mInputs.dimB()},
                               std::vector<unsigned int>());

    if (!mLockRandom) {
        generateRegions(mGridX, mInputs[0].dimX(), mOutputs.dimX());
        generateRegions(mGridY, mInputs[0].dimY(), mOutputs.dimY());
    }

    const unsigned int size = mInputs.dimB() * getNbOutputs();

    const Tensor<Float_T>& input0 = tensor_cast<Float_T>(mInputs[0]);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < getNbOutputs(); ++output) {
            for (unsigned int oy = 0; oy < mOutputs.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < mOutputs.dimX(); ++ox) {
                    if (mPoolNbChannels[output] == 0)
                        continue; // No connection to this output...

                    // For each output, compute the pool value
                    Float_T poolValue =
                        -std::numeric_limits<Float_T>::infinity();
                    unsigned int channelMax = 0;
                    unsigned int ixMax = 0;
                    unsigned int iyMax = 0;

                    for (unsigned int channel = 0; channel < getNbChannels();
                         ++channel) {
                        if (!isConnection(channel, output))
                            continue;

                        const unsigned int ixStart = (ox > 0) ? mGridX[ox - 1]
                                                              : 0;
                        const unsigned int iyStart = (oy > 0) ? mGridY[oy - 1]
                                                              : 0;
                        unsigned int ixStop = mGridX[ox];
                        unsigned int iyStop = mGridY[oy];

                        if (!mOverlapping) {
                            --ixStop;
                            --iyStop;
                        }

                        if (ox == mOutputs.dimX() - 1)
                            ixStop = input0.dimX() - 1;

                        if (oy == mOutputs.dimY() - 1)
                            iyStop = input0.dimY() - 1;

                        for (unsigned int iy = iyStart; iy <= iyStop; ++iy) {
                            for (unsigned int ix = ixStart; ix <= ixStop;
                                 ++ix) {
                                if (input0(ix, iy, channel, batchPos)
                                    > poolValue) {
                                    poolValue
                                        = input0(ix, iy, channel, batchPos);

                                    channelMax = channel;
                                    ixMax = ix;
                                    iyMax = iy;
                                }
                            }
                        }
                    }

                    if (!inference) {
// For this output node, take the max. input across all connected channels
// Critical section needed because channelMax can be the same for several
// outputs
#pragma omp critical(FMPCell_Frame__propagate)
                        mInputsBackProp(ixMax, iyMax, channelMax, batchPos)
                            .push_back(ox + oy * mOutputs.dimX()
                                       + output * mOutputs.dimX()
                                         * mOutputs.dimY());
                    }

                    // Compute the output signal
                    mOutputs(ox, oy, output, batchPos) = poolValue;
                }
            }
        }
    }

    Cell_Frame<Float_T>::propagate(inference);
    mDiffInputs.clearValid();
}

void N2D2::FMPCell_Frame::backPropagate()
{
    if (mDiffOutputs[0].empty() || !mDiffInputs.isValid())
        return;

    Cell_Frame<Float_T>::backPropagate();

    const unsigned int size = mInputs.dimB() * getNbChannels();

    Tensor<Float_T> diffOutput0 = (mDiffOutputs[0].isValid())
        ? tensor_cast<Float_T>(mDiffOutputs[0])
        : tensor_cast_nocopy<Float_T>(mDiffOutputs[0]);

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (mInputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
        {
            const bool isValid = mDiffOutputs[0].isValid();

            for (unsigned int iy = 0; iy < mInputs[0].dimY(); ++iy) {
                for (unsigned int ix = 0; ix < mInputs[0].dimX(); ++ix) {
                    Float_T gradient = 0.0;

                    // In max pooling, the unit which was chosen as the max
                    // receives all the error since very small changes
                    // in input would perturb the result only through that unit
                    for (std::vector<unsigned int>::const_iterator it
                         = mInputsBackProp(ix, iy, channel, batchPos).begin(),
                         itEnd
                         = mInputsBackProp(ix, iy, channel, batchPos).end();
                         it != itEnd;
                         ++it) {
                        gradient += mDiffInputs(*it, batchPos);
                    }

                    diffOutput0(ix, iy, channel, batchPos)
                        = gradient + isValid
                                     * diffOutput0(ix, iy, channel, batchPos);
                }
            }
        }
    }

    mDiffOutputs[0] = diffOutput0;
    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

void N2D2::FMPCell_Frame::update()
{
    Cell_Frame<float>::update();
}

void N2D2::FMPCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&FMPCell_Frame::propagate, this, false),
                  std::bind(&FMPCell_Frame::backPropagate, this),
                  true);

    mLockRandom = true;

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

    mLockRandom = false;
}

void N2D2::FMPCell_Frame::logRegions(const std::string& fileName) const
{
    Tensor<Float_T> regions({mInputs[0].dimX(), mInputs[0].dimY()}, 0.0);

    for (unsigned int oy = 0; oy < mOutputs.dimY(); ++oy) {
        for (unsigned int ox = 0; ox < mOutputs.dimX(); ++ox) {
            const unsigned int ixStart = (ox > 0) ? mGridX[ox - 1] : 0;
            const unsigned int iyStart = (oy > 0) ? mGridY[oy - 1] : 0;
            unsigned int ixStop = mGridX[ox];
            unsigned int iyStop = mGridY[oy];

            if (!mOverlapping) {
                --ixStop;
                --iyStop;
            }

            if (ox == mOutputs.dimX() - 1)
                ixStop = mInputs[0].dimX() - 1;

            if (oy == mOutputs.dimY() - 1)
                iyStop = mInputs[0].dimY() - 1;

            for (unsigned int iy = iyStart; iy <= iyStop; ++iy) {
                for (unsigned int ix = ixStart; ix <= ixStop; ++ix)
                    regions(ix, iy) += ((ox % 2) != (oy % 2)) ? 0 : 1;
            }
        }
    }

    StimuliProvider::logData(fileName, regions);
}

void N2D2::FMPCell_Frame::generateRegions(std::vector<unsigned int>& grid,
                                          unsigned int sizeIn,
                                          unsigned int sizeOut)
{
    grid.resize(sizeOut);

    if (mPseudoRandom) {
        // Compute the true scaling ratio
        // This is important to obtain the correct range
        const double scalingRatio = sizeIn / (double)sizeOut;
        const double u = Random::randUniform(0.0, 1.0, Random::OpenInterval);

        for (unsigned int i = 0; i < sizeOut; ++i)
            grid[i] = (unsigned int)std::ceil(scalingRatio * (i + u));
    } else {
        const unsigned int nb2 = sizeIn - sizeOut;
        // const unsigned int nb1 = 2*sizeOut - sizeIn;
        // assert(nb1 + nb2 == sizeOut);

        std::fill(grid.begin(), grid.begin() + nb2, 2);
        std::fill(grid.begin() + nb2, grid.end(), 1);

        // Random shuffle
        for (int i = grid.size() - 1; i > 0; --i)
            std::swap(grid[i], grid[Random::randUniform(0, i)]);

        for (unsigned int i = 1; i < grid.size(); ++i)
            grid[i] += grid[i - 1];
    }
}
