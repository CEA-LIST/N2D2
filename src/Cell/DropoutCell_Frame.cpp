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

N2D2::Registrar<N2D2::DropoutCell>
N2D2::DropoutCell_Frame::mRegistrar("Frame",
                                         N2D2::DropoutCell_Frame::create);

N2D2::DropoutCell_Frame::DropoutCell_Frame(const std::string& name,
                                                     unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      DropoutCell(name, nbOutputs),
      Cell_Frame(name, nbOutputs)
{
    // ctor
}

void N2D2::DropoutCell_Frame::initialize()
{
    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("DropoutCell_Frame::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for DropoutCell "
                                     + mName);
        }
    }

    mMask.resize(mOutputs.dimX(),
                 mOutputs.dimY(),
                 mOutputs.dimZ(),
                 mOutputs.dimB());
}

void N2D2::DropoutCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDToH();

    unsigned int offset = 0;

    if (inference) {
        if (mInputs.size() == 1)
            std::copy(mInputs[0].begin(), mInputs[0].end(), mOutputs.begin());
        else {
            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                unsigned int outputOffset = offset;
                unsigned int inputOffset = 0;

                for (unsigned int batchPos = 0; batchPos < mInputs.dimB();
                    ++batchPos)
                {
                    std::copy(mInputs[k].begin() + inputOffset,
                              mInputs[k].begin() + inputOffset
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
                        ? mInputs[k](index + inputOffset)
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

void N2D2::DropoutCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mDiffOutputs[k].isValid())
            throw std::runtime_error(
                "Cannot blend gradient from a Dropout cell");

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
                mDiffOutputs[k](index + inputOffset) = (mMask(outputIndex))
                    ? mDiffInputs(outputIndex)
                    : 0.0;
            }

            outputOffset += mOutputs.dimX() * mOutputs.dimY()
                            * mInputs.dimZ();
            inputOffset += mOutputs.dimX() * mOutputs.dimY()
                           * mInputs[k].dimZ();
        }

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::DropoutCell_Frame::update()
{
}

N2D2::DropoutCell_Frame::~DropoutCell_Frame()
{
    //dtor
}
