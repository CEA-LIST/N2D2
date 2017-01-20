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

#include "Cell/Cell_Spike.hpp"

#include "Cell/NodeIn.hpp"
#include "Cell/NodeOut.hpp"

N2D2::Cell_Spike::Cell_Spike(Network& net,
                             const std::string& name,
                             unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      NetworkObserver(net),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mIncomingDelay(this, "IncomingDelay", 1 * TimePs, 100 * TimeFs),
      mNbChannels(0),
      mNet(net)
{
    // ctor
}

void N2D2::Cell_Spike::addInput(StimuliProvider& sp,
                                unsigned int channel,
                                unsigned int x0,
                                unsigned int y0,
                                unsigned int width,
                                unsigned int height,
                                const std::vector<bool>& mapping)
{
    Environment* env = dynamic_cast<Environment*>(&sp);

    if (env == NULL)
        throw std::runtime_error(
            "Cell_Spike::addInput(): Spiking models require Environment");

    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    // Define input-output sizes
    setInputsSize(width, height);
    mNbChannels += 1;
    setOutputsSize();

    // Define input-output connections
    if (!mapping.empty() && mapping.size() != mNbOutputs)
        throw std::runtime_error("Cell_Spike::addInput(): mapping length must "
                                 "be equal to the number of outputs");

    mMaps.resize(mNbOutputs, mNbChannels);

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        mMaps(output, mNbChannels - 1) = (!mapping.empty()) ? mapping[output]
                                                            : true;
    }

    mInputs.reserve(mNbChannels * mChannelsWidth * mChannelsHeight);

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            mInputs.push_back(new NodeIn(
                mNet,
                *this,
                (mNbChannels - 1) + (x + width * y)
                                    / (mChannelsWidth * mChannelsHeight)));
            mInputs.back()->addLink(env->getNode(channel, x0 + x, y0 + y));
        }
    }

    populateOutputs();
}

void N2D2::Cell_Spike::addInput(StimuliProvider& sp,
                                unsigned int x0,
                                unsigned int y0,
                                unsigned int width,
                                unsigned int height,
                                const Matrix<bool>& mapping)
{
    const unsigned int nbChannels = sp.getNbChannels();

    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    if (!mapping.empty() && mapping.rows() != nbChannels)
        throw std::runtime_error("Cell_Spike::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "filters");

    for (unsigned int channel = 0; channel < nbChannels; ++channel) {
        // Cell_Spike:: prefix needed to avoid ambiguity with Transcode model
        Cell_Spike::addInput(sp,
                             channel,
                             x0,
                             y0,
                             width,
                             height,
                             (mapping.empty()) ? std::vector<bool>()
                                               : mapping.row(channel));
    }
}

void N2D2::Cell_Spike::addInput(Cell* cell, const Matrix<bool>& mapping)
{
    Cell_Spike* cellSpike = dynamic_cast<Cell_Spike*>(cell);

    if (cellSpike == NULL)
        throw std::runtime_error(
            "Cell_Spike::addInput(): cannot mix Spike and Frame models");

    // Define input-output sizes
    setInputsSize(cellSpike->mOutputsWidth, cellSpike->mOutputsHeight);
    mNbChannels += cellSpike->mNbOutputs;
    setOutputsSize();

    // Define input-output connections
    if (!mapping.empty() && mapping.rows() != cellSpike->mNbOutputs)
        throw std::runtime_error("Cell_Spike::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "channels");

    mMaps.resize(mNbOutputs, mNbChannels);
    const unsigned int channelOffset = mNbChannels - cellSpike->mNbOutputs;

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < cellSpike->mNbOutputs;
             ++channel) {
            mMaps(output, channelOffset + channel)
                = (!mapping.empty()) ? mapping(channel, output) : true;
        }
    }

    mInputs.reserve(mNbChannels * mChannelsWidth * mChannelsHeight);

    for (unsigned int index = 0,
                      size = cellSpike->mNbOutputs * cellSpike->mOutputsHeight
                             * cellSpike->mOutputsWidth;
         index < size;
         ++index) {
        mInputs.push_back(
            new NodeIn(mNet,
                       *this,
                       (mNbChannels - cellSpike->mNbOutputs)
                       + index / (mChannelsWidth * mChannelsHeight)));
        mInputs.back()->addLink(cellSpike->mOutputs(index));
    }

    populateOutputs();
}

void N2D2::Cell_Spike::addInput(Cell* cell,
                                unsigned int x0,
                                unsigned int y0,
                                unsigned int width,
                                unsigned int height)
{
    Cell_Spike* cellSpike = dynamic_cast<Cell_Spike*>(cell);

    if (cellSpike == NULL)
        throw std::runtime_error(
            "Cell_Spike::addInput(): cannot mix Spike and Frame models");

    if (width == 0)
        width = cellSpike->mOutputsWidth - x0;
    if (height == 0)
        height = cellSpike->mOutputsHeight - y0;

    // Define input-output sizes
    setInputsSize(width, height);
    mNbChannels += cellSpike->mNbOutputs;
    setOutputsSize();

    // Define input-output connections
    mMaps.resize(mNbOutputs, mNbChannels, true);

    mInputs.reserve(mNbChannels * mChannelsWidth * mChannelsHeight);

    for (unsigned int output = 0; output < cellSpike->mNbOutputs; ++output) {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                mInputs.push_back(
                    new NodeIn(mNet,
                               *this,
                               (mNbChannels - cellSpike->mNbOutputs)
                               + (output * height * width + (x + width * y))
                                 / (width * height)));
                mInputs.back()->addLink(
                    cellSpike->mOutputs(x0 + x, y0 + y, output, 0));
            }
        }
    }

    populateOutputs();
}

void N2D2::Cell_Spike::populateOutputs()
{
    if (mOutputs.empty()) {
        mOutputs.resize(mOutputsWidth, mOutputsHeight, mNbOutputs, 1, NULL);

        for (unsigned int output = 0; output < mNbOutputs; ++output) {
            for (unsigned int y = 0; y < mOutputsHeight; ++y) {
                for (unsigned int x = 0; x < mOutputsWidth; ++x)
                    mOutputs(x, y, output, 0)
                        = new NodeOut(mNet,
                                      *this,
                                      output,
                                      1.0,
                                      output / (double)mNbOutputs,
                                      x,
                                      y);
            }
        }
    }
}

N2D2::Cell_Spike::~Cell_Spike()
{
    // dtor
    std::for_each(mInputs.begin(), mInputs.end(), Utils::Delete());
    std::for_each(mOutputs.begin(), mOutputs.end(), Utils::Delete());
}
