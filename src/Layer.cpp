/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

#include "Layer.hpp"

N2D2::Layer::Layer(Network& net, unsigned int nbCells) //: mNet(net)
{
    // ctor
    mCells.reserve(nbCells);

    for (unsigned int i = 0; i < nbCells; ++i)
        mCells.push_back(new Xcell(net));

    mInputOffset = 0;
}

void N2D2::Layer::setInputOffset(unsigned int offset)
{
    mInputOffset = offset;
}

void N2D2::Layer::addLateralInhibition()
{
    for (std::vector<Xcell*>::const_iterator it = mCells.begin(),
                                             itEnd = mCells.end();
         it != itEnd;
         ++it) {
        for (std::vector<Xcell*>::const_iterator itInner = mCells.begin(),
                                                 itInnerEnd = mCells.end();
             itInner != itInnerEnd;
             ++itInner) {
            if (it != itInner)
                (*it)->addLateralInhibition(*(*itInner));
        }
    }
}

void N2D2::Layer::addInput(Layer& layer)
{
    for (std::vector<Xcell*>::iterator it = layer.mCells.begin(),
                                       itEnd = layer.mCells.end();
         it != itEnd;
         ++it) {
        std::for_each(mCells.begin(),
                      mCells.end(),
                      std::bind(static_cast
                                <void (Xcell::*)(Xcell&)>(&Xcell::addInput),
                                std::placeholders::_1,
                                std::ref(*(*it))));
    }
}

void N2D2::Layer::addInput(Xcell& cell)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(Xcell&)>(&Xcell::addInput),
                            std::placeholders::_1,
                            std::ref(cell)));
}

void N2D2::Layer::addInput(Layer& layer,
                           unsigned int sizeX,
                           unsigned int sizeY,
                           unsigned int cellsX,
                           unsigned int cellsY,
                           unsigned int prevCellsX,
                           unsigned int prevCellsY)
{
    if (cellsX == 0 || cellsY == 0)
        throw std::runtime_error("The number of cells in X and Y must be > 0");

    if (mInputOffset + cellsX * cellsY > mCells.size())
        throw std::runtime_error("Layer size too small");

    if (prevCellsX * prevCellsY > layer.mCells.size())
        throw std::runtime_error("Previous layer size too small");

    for (unsigned int x = 0; x < cellsX; ++x) {
        for (unsigned int y = 0; y < cellsY; ++y) {
            const unsigned int prevX
                = (cellsX > 1) ? x * (prevCellsX - sizeX) / (cellsX - 1) : 0;
            const unsigned int prevY
                = (cellsY > 1) ? y * (prevCellsY - sizeY) / (cellsY - 1) : 0;

            for (unsigned int ox = 0; ox < sizeX; ++ox) {
                for (unsigned int oy = 0; oy < sizeY; ++oy)
                    mCells[mInputOffset + x + cellsX * y]->addInput(
                        *layer.mCells
                             [(prevX + ox) + prevCellsX * (prevY + oy)]);
            }
        }
    }
}

void N2D2::Layer::addInput(Environment& env,
                           unsigned int channel,
                           unsigned int sizeX,
                           unsigned int sizeY,
                           unsigned int cellsX,
                           unsigned int cellsY)
{
    if (cellsX == 0 || cellsY == 0)
        throw std::runtime_error("The number of cells in X and Y must be > 0");

    if (mInputOffset + cellsX * cellsY > mCells.size())
        throw std::runtime_error("Layer size too small");

    const unsigned int envSizeX = env.getSizeX();
    const unsigned int envSizeY = env.getSizeY();

    for (unsigned int x = 0; x < cellsX; ++x) {
        for (unsigned int y = 0; y < cellsY; ++y) {
            const unsigned int offsetX
                = (cellsX > 1) ? x * (envSizeX - sizeX) / (cellsX - 1) : 0;
            const unsigned int offsetY
                = (cellsY > 1) ? y * (envSizeY - sizeY) / (cellsY - 1) : 0;

            mCells[mInputOffset + x + cellsX * y]
                ->addInput(env, channel, offsetX, offsetY, sizeX, sizeY);
        }
    }
}

void N2D2::Layer::addInput(Environment& env,
                           unsigned int sizeX,
                           unsigned int sizeY,
                           unsigned int cellsX,
                           unsigned int cellsY)
{
    for (unsigned int channel = 0, nbChannels = env.getNbChannels();
         channel < nbChannels;
         ++channel)
        addInput(env, channel, sizeX, sizeY, cellsX, cellsY);
}

void N2D2::Layer::addMultiscaleInput(HeteroEnvironment& env,
                                     unsigned int sizeX,
                                     unsigned int sizeY,
                                     unsigned int cellsX,
                                     unsigned int cellsY)
{
    for (unsigned int map = 0, size = env.size(); map < size; ++map)
        addInput(*(env[map]), sizeX, sizeY, cellsX, cellsY);
}

void N2D2::Layer::readActivity(const std::string& fileName)
{
    std::for_each(
        mCells.begin(),
        mCells.end(),
        std::bind(&Xcell::readActivity, std::placeholders::_1, fileName));
}

void N2D2::Layer::save(const std::string& dirName) const
{
    std::for_each(
        mCells.begin(),
        mCells.end(),
        std::bind(&Xcell::save, std::placeholders::_1, dirName, false));
}

void N2D2::Layer::load(const std::string& dirName)
{
    std::for_each(
        mCells.begin(),
        mCells.end(),
        std::bind(&Xcell::load, std::placeholders::_1, dirName, false));
}

void
N2D2::Layer::logState(const std::string& dirName, bool append, bool plot) const
{
    std::for_each(
        mCells.begin(),
        mCells.end(),
        std::bind(
            &Xcell::logState, std::placeholders::_1, dirName, append, plot));
}

void N2D2::Layer::logStats(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create stats log file: "
                                 + fileName);

    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(std::ofstream & dataFile) const>(
                                &Xcell::logStats),
                            std::placeholders::_1,
                            std::ref(data)));
}

void N2D2::Layer::clearStats()
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(&Xcell::clearStats, std::placeholders::_1));
}

const std::vector<N2D2::NodeNeuron*> N2D2::Layer::getNeurons() const
{
    std::vector<NodeNeuron*> neurons;

    for (std::vector<Xcell*>::const_iterator it = mCells.begin(),
                                             itEnd = mCells.end();
         it != itEnd;
         ++it) {
        const std::vector<NodeNeuron*>& cellNeurons = (*it)->getNeurons();
        neurons.insert(neurons.end(), cellNeurons.begin(), cellNeurons.end());
    }

    return neurons;
}

cv::Mat N2D2::Layer::reconstructPattern(unsigned int cell,
                                        unsigned int neuron,
                                        bool normalize,
                                        bool multiLayer) const
{
    return mCells.at(cell)->reconstructPattern(neuron, normalize, multiLayer);
}

void N2D2::Layer::reconstructPatterns(const std::string& dirName,
                                      bool normalize,
                                      bool multiLayer) const
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(&Xcell::reconstructPatterns,
                            std::placeholders::_1,
                            dirName,
                            normalize,
                            multiLayer));
}

std::string N2D2::Layer::getNeuronsParameter(const std::string& name) const
{
    const std::string value = mCells.front()->getNeuronsParameter(name);

    if (std::find_if(
            mCells.begin(),
            mCells.end(),
            std::bind(
                std::not_equal_to<std::string>(),
                value,
                std::bind(static_cast
                          <std::string (Xcell::*)(const std::string&)const>(
                              &Xcell::getParameter),
                          std::placeholders::_1,
                          name))) != mCells.end()) {
        throw std::runtime_error(
            "Different values within the cell for neurons parameter: " + name);
    }

    return value;
}

void N2D2::Layer::setNeuronsParameters(const std::map
                                       <std::string, std::string>& params,
                                       bool ignoreUnknown)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(&Xcell::setNeuronsParameters,
                            std::placeholders::_1,
                            params,
                            ignoreUnknown));
}

void N2D2::Layer::loadCellsParameters(const std::string& fileName,
                                      bool ignoreNotExists,
                                      bool ignoreUnknown)
{
    if (!std::ifstream(fileName.c_str()).good()) {
        if (ignoreNotExists)
            std::cout << "Notice: Could not open configuration file: "
                      << fileName << std::endl;
        else
            throw std::runtime_error("Could not open configuration file: "
                                     + fileName);
    } else {
        // ignoreNotExists is set to false in the following calls, because the
        // file is supposed to exist and be readable.
        // Otherwise, something is really wrong and it's probably better to
        // throw an exception!
        std::for_each(mCells.begin(),
                      mCells.end(),
                      std::bind(&Xcell::loadParameters,
                                std::placeholders::_1,
                                fileName,
                                false,
                                ignoreUnknown));
    }
}

void N2D2::Layer::saveCellsParameters(const std::string& fileName) const
{
    mCells.front()->saveParameters(fileName);
}

void N2D2::Layer::loadNeuronsParameters(const std::string& fileName,
                                        bool ignoreNotExists,
                                        bool ignoreUnknown)
{
    if (!std::ifstream(fileName.c_str()).good()) {
        if (ignoreNotExists)
            std::cout << "Notice: Could not open configuration file: "
                      << fileName << std::endl;
        else
            throw std::runtime_error("Could not open configuration file: "
                                     + fileName);
    } else {
        // ignoreNotExists is set to false in the following calls, because the
        // file is supposed to exist and be readable.
        // Otherwise, something is really wrong and it's probably better to
        // throw an exception!
        std::for_each(mCells.begin(),
                      mCells.end(),
                      std::bind(&Xcell::loadNeuronsParameters,
                                std::placeholders::_1,
                                fileName,
                                false,
                                ignoreUnknown));
    }
}

void N2D2::Layer::saveNeuronsParameters(const std::string& fileName) const
{
    mCells.front()->saveNeuronsParameters(fileName);
}

void N2D2::Layer::copyCellsParameters(const Xcell& from)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(&Xcell::copyParameters,
                            std::placeholders::_1,
                            std::ref(from)));
}

void N2D2::Layer::copyCellsParameters(const Layer& from)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(&Xcell::copyParameters,
                            std::placeholders::_1,
                            std::ref(*from.mCells.front())));
}

void N2D2::Layer::copyNeuronsParameters(const NodeNeuron& from)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast<void (Xcell::*)(const NodeNeuron&)>(
                                &Xcell::copyNeuronsParameters),
                            std::placeholders::_1,
                            std::ref(from)));
}

void N2D2::Layer::copyNeuronsParameters(const Xcell& from)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast<void (Xcell::*)(const Xcell&)>(
                                &Xcell::copyNeuronsParameters),
                            std::placeholders::_1,
                            std::ref(from)));
}

void N2D2::Layer::copyNeuronsParameters(const Layer& from)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast<void (Xcell::*)(const Xcell&)>(
                                &Xcell::copyNeuronsParameters),
                            std::placeholders::_1,
                            std::ref(*from.mCells.front())));
}

N2D2::Layer::~Layer()
{
    // dtor
    std::for_each(mCells.begin(), mCells.end(), Utils::Delete());
}
