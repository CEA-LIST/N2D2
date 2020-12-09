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

#include "Xnet/NodeEnv.hpp"
#include "Cell/NodeIn.hpp"
#include "Cell/NodeOut.hpp"
#include "DeepNet.hpp"
#include "Xnet/Environment.hpp"
#include "StimuliProvider.hpp"

N2D2::Cell_Spike::Cell_Spike(Network& net, const DeepNet& deepNet, 
                             const std::string& name,
                             unsigned int nbOutputs)
    : Cell(deepNet, name, nbOutputs),
      NetworkObserver(net),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mIncomingDelay(this, "IncomingDelay", 1 * TimePs, 100 * TimeFs),
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
                                const Tensor<bool>& mapping)
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
    setInputsDims({width, height, 1U});
    setOutputsDims();

    // Define input-output connections
    if (!mapping.empty() && (mapping.nbDims() != 1
                            || mapping.size() != getNbOutputs()))
    {
        std::ostringstream msgStr;
        msgStr << "Cell_Spike::addInput(): mapping must be 1D and its size ("
            << mapping.size() << ") must be equal to the number of outputs ("
            << getNbOutputs() << ")";

        throw std::runtime_error(msgStr.str());
    }

    // Use push_back() instead of append() here because the argument is 1D
    mMapping.push_back((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs()}, true));

    mInputs.reserve(getInputsSize());

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            mInputs.push_back(new NodeIn(
                mNet,
                *this,
                (getNbChannels() - 1) + (x + width * y)
                                    / (mInputsDims[0] * mInputsDims[1])));
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
                                const Tensor<bool>& mapping)
{
    const unsigned int nbChannels = sp.getNbChannels();

    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    if (!mapping.empty() && mapping.dimY() != nbChannels) {
        std::ostringstream msgStr;
        msgStr << "Cell_Spike::addInput(): number of mapping "
            "rows (" << mapping.dimY() << ") must be equal to the number of "
            "input filters (" << nbChannels << ")";

        throw std::runtime_error(msgStr.str());
    }

    for (unsigned int channel = 0; channel < nbChannels; ++channel) {
        // Cell_Spike:: prefix needed to avoid ambiguity with Transcode model
        Cell_Spike::addInput(sp,
                             channel,
                             x0,
                             y0,
                             width,
                             height,
                             (mapping.empty()) ? Tensor<bool>()
                                               : mapping[channel]);
    }
}

void N2D2::Cell_Spike::addInput(Cell* cell, const Tensor<bool>& mapping)
{
    Cell_Spike* cellSpike = dynamic_cast<Cell_Spike*>(cell);

    if (cellSpike == NULL)
        throw std::runtime_error(
            "Cell_Spike::addInput(): cannot mix Spike and Frame models");

    // Define input-output sizes
    setInputsDims(cellSpike->getOutputsDims());
    setOutputsDims();

    // Define input-output connections
    const unsigned int cellNbOutputs = cellSpike->getNbOutputs();

    if (!mapping.empty() && mapping.dimY() != cellNbOutputs) {
        std::ostringstream msgStr;
        msgStr << "Cell_Spike::addInput(): number of mapping "
            "rows (" << mapping.dimY() << ") must be equal to the number of "
            "input filters (" << cellNbOutputs << ")";

        throw std::runtime_error(msgStr.str());
    }

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), cellNbOutputs}, true));

    mInputs.reserve(getInputsSize());

    for (unsigned int index = 0, size = cellSpike->getOutputsSize();
        index < size; ++index)
    {
        mInputs.push_back(
            new NodeIn(mNet,
                       *this,
                       (getNbChannels() - cellSpike->getNbOutputs())
                       + index / (mInputsDims[0] * mInputsDims[1])));
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
        width = cellSpike->getOutputsDim(0) - x0;
    if (height == 0)
        height = cellSpike->getOutputsDim(1) - y0;

    // Define input-output sizes
    const unsigned int cellNbOutputs = cellSpike->getNbOutputs();

    setInputsDims({width, height, cellNbOutputs});
    setOutputsDims();

    // Define input-output connections
    mMapping.resize({getNbOutputs(), getNbChannels()}, true);

    mInputs.reserve(getInputsSize());

    for (unsigned int output = 0; output < cellNbOutputs; ++output) {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                mInputs.push_back(
                    new NodeIn(mNet,
                               *this,
                               (getNbChannels() - cellNbOutputs)
                               + (output * height * width + (x + width * y))
                                 / (width * height)));
                mInputs.back()->addLink(
                    cellSpike->mOutputs(x0 + x, y0 + y, output, 0));
            }
        }
    }

    populateOutputs();
}

void N2D2::Cell_Spike::clearInputs() {
    throw std::runtime_error("Cell_Spike::clearInputs(): not supported.");
}

void N2D2::Cell_Spike::populateOutputs()
{
    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(1);

        mOutputs.resize(outputsDims, NULL);

        for (unsigned int output = 0; output < getNbOutputs(); ++output) {
            for (unsigned int y = 0; y < mOutputsDims[1]; ++y) {
                for (unsigned int x = 0; x < mOutputsDims[0]; ++x)
                    mOutputs(x, y, output, 0)
                        = new NodeOut(mNet,
                                      *this,
                                      output,
                                      1.0,
                                      output / (double)getNbOutputs(),
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
