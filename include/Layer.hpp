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

#ifndef N2D2_LAYER_H
#define N2D2_LAYER_H

#include <algorithm>
#include <string>
#include <vector>

#include "HeteroEnvironment.hpp"
#include "Network.hpp"
#include "Xcell.hpp"

namespace N2D2 {
/**
 * A Layer is a group of Xcells and it is usefull to create non fully connected
 * network, because each cell in the layer can be
 * connected to a different part of the previous layer (or the environment) in a
 * regular fashion, like in a grid pattern for
 * example. Most of the methods of Layer are just calling the corresponding
 * method of Xcell for each cells in the layer.
*/
class Layer {
public:
    Layer(Network& net, unsigned int nbCells = 1);
    template <class T>
    void populate(unsigned int nbNeurons,
                  Xcell::InhibitionTopology inhibition = Xcell::Symmetric);
    void setInputOffset(unsigned int offset = 0);

    /**
     * Add lateral inhibition between all the Xcell in the Layer.
    */
    void addLateralInhibition();

    /**
     * Connect this layer to a layer (fully connected).
     *
     * @param layer Layer to connect to
    */
    void addInput(Layer& layer);

    /**
     * Connect this layer to a Xcell (fully connected).
     *
     * @param cell Xcell to connect to
    */
    void addInput(Xcell& cell);

    /**
     * Connect this layer to a layer with a 2D topology.
     *
     * Because a layer does not contain any topologycal information (it is only
     *a vector of Xcell), one has to specify in
     * how many rows and columns the Xcells from the previous layer are
     *organized, with the parameters @p prevCellsY and
     * @p prevCellsX respectively.
     *
     * @param layer Previous layer to connect to
     * @param sizeX Width of a Xcell in this layer (in number of Xcells from the
     *previous layer)
     * @param sizeY Height of a Xcell in this layer (in number of Xcells from
     *the previous layer)
     * @param cellsX Number of Xcell in a row in this layer
     * @param cellsY Number of Xcell in a column in this layer
     * @param prevCellsX Number of Xcells constituting a row in the previous
     *layer
     * @param prevCellsY Number of Xcells constituting a column in the previous
     *layer
     *
     * @exception std::runtime_error This layer must contain at least @p
     *cellsX*@p cellsY Xcells.
    */
    void addInput(Layer& layer,
                  unsigned int sizeX,
                  unsigned int sizeY,
                  unsigned int cellsX,
                  unsigned int cellsY,
                  unsigned int prevCellsX,
                  unsigned int prevCellsY);

    /**
     * Connect this layer to a single channel from one map of the Environment.
     *
     * @param env The Environment
     * @param channel Channel in the @p map to connect to
     * @param sizeX Width of a Xcell (in number of spiking pixels)
     * @param sizeY Height of a Xcell (in number of spiking pixels)
     * @param cellsX Number of Xcells in a row in this layer
     * @param cellsY Number of Xcells in a column in this layer
     *
     * @exception std::runtime_error This layer must contain at least @p
     *cellsX*@p cellsY Xcells.
    */
    void addInput(Environment& env,
                  unsigned int channel,
                  unsigned int sizeX,
                  unsigned int sizeY,
                  unsigned int cellsX,
                  unsigned int cellsY);

    /**
     * Connect this layer to all the channels from one map of the Environment.
     *
     * @param env The Environment
     * @param sizeX Width of a Xcell (in number of spiking pixels)
     * @param sizeY Height of a Xcell (in number of spiking pixels)
     * @param cellsX Number of Xcells in a row in this layer
     * @param cellsY Number of Xcells in a column in this layer
     *
     * @exception std::runtime_error This layer must contain at least @p
     *cellsX*@p cellsY Xcells.
    */
    void addInput(Environment& env,
                  unsigned int sizeX,
                  unsigned int sizeY,
                  unsigned int cellsX,
                  unsigned int cellsY);

    /**
     * Connect this layer to all the channels of all the maps of the Environment
     * (i.e. fully connect the layer to the Environment).
     *
     * @param env The Environment
     * @param sizeX Width of a Xcell (in number of spiking pixels)
     * @param sizeY Height of a Xcell (in number of spiking pixels)
     * @param cellsX Number of Xcells in a row in this layer
     * @param cellsY Number of Xcells in a column in this layer
     *
     * @exception std::runtime_error This layer must contain at least @p
     *cellsX*@p cellsY Xcells.
    */
    void addMultiscaleInput(HeteroEnvironment& env,
                            unsigned int sizeX,
                            unsigned int sizeY,
                            unsigned int cellsX,
                            unsigned int cellsY);
    void readActivity(const std::string& fileName);
    void save(const std::string& dirName) const;
    void load(const std::string& dirName);
    void logState(const std::string& dirName,
                  bool append = false,
                  bool plot = true) const;
    void logStats(const std::string& fileName) const;
    void clearStats();
    const std::vector<Xcell*>& getCells() const
    {
        return mCells;
    };
    const std::vector<NodeNeuron*> getNeurons() const;
    cv::Mat reconstructPattern(unsigned int cell,
                               unsigned int neuron,
                               bool normalize = false,
                               bool multiLayer = true) const;
    void reconstructPatterns(const std::string& dirName,
                             bool normalize = false,
                             bool multiLayer = true) const;

    std::string getNeuronsParameter(const std::string& name) const;
    template <class T> T getNeuronsParameter(const std::string& name) const;
    template <class T> void setCellsParameter(const std::string& name, T value);
    template <class T>
    void setCellsParameter(const std::string& name, T mean, Percent relStdDev);
    template <class T>
    void setCellsParameter(const std::string& name, T mean, double stdDev);
    template <class T>
    void setNeuronsParameter(const std::string& name, T value);
    template <class T>
    void
    setNeuronsParameter(const std::string& name, T mean, Percent relStdDev);
    template <class T>
    void setNeuronsParameter(const std::string& name, T mean, double stdDev);
    template <class T>
    void setNeuronsParameterSpread(const std::string& name,
                                   Percent relStdDev = Percent(0));
    template <class T>
    void setNeuronsParameterSpread(const std::string& name, double stdDev);
    void setNeuronsParameters(const std::map<std::string, std::string>& params,
                              bool ignoreUnknown = false);
    void loadCellsParameters(const std::string& fileName,
                             bool ignoreNotExists = false,
                             bool ignoreUnknown = false);
    void saveCellsParameters(const std::string& fileName) const;
    void loadNeuronsParameters(const std::string& fileName,
                               bool ignoreNotExists = false,
                               bool ignoreUnknown = false);
    void saveNeuronsParameters(const std::string& fileName) const;
    void copyCellsParameters(const Xcell& from);
    void copyCellsParameters(const Layer& from);
    void copyNeuronsParameters(const NodeNeuron& from);
    void copyNeuronsParameters(const Xcell& from);
    void copyNeuronsParameters(const Layer& from);
    virtual ~Layer();

private:
    //Network& mNet; ///< Reference to the Network attached to this object.
    /// Vector of pointers to all the Xcells contained in this layer
    std::vector<Xcell*> mCells;
    unsigned int mInputOffset;
};
}

template <class T>
void N2D2::Layer::populate(unsigned int nbNeurons,
                           Xcell::InhibitionTopology inhibition)
{
    std::for_each(
        mCells.begin(),
        mCells.end(),
        std::bind(
            &Xcell::populate<T>, std::placeholders::_1, nbNeurons, inhibition));
}

template <class T>
void N2D2::Layer::setCellsParameter(const std::string& name, T value)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast<void (Xcell::*)(const std::string&, T)>(
                                &Xcell::setParameter<T>),
                            std::placeholders::_1,
                            name,
                            value));
}

template <class T>
void N2D2::Layer::setCellsParameter(const std::string& name,
                                    T mean,
                                    Percent relStdDev)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(const std::string&, T, Percent)>(
                                &Xcell::setParameter<T>),
                            std::placeholders::_1,
                            name,
                            mean,
                            relStdDev));
}

template <class T>
void
N2D2::Layer::setCellsParameter(const std::string& name, T mean, double stdDev)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(const std::string&, T, double)>(
                                &Xcell::setParameter<T>),
                            std::placeholders::_1,
                            name,
                            mean,
                            stdDev));
}

template <class T>
T N2D2::Layer::getNeuronsParameter(const std::string& name) const
{
    const T value = mCells.front()->getNeuronsParameter<T>(name);

    if (std::find_if(mCells.begin(),
                     mCells.end(),
                     std::bind(std::not_equal_to<T>(),
                               value,
                               std::bind(&Xcell::getNeuronsParameter<T>,
                                         std::placeholders::_1,
                                         name))) != mCells.end()) {
        throw std::runtime_error(
            "Different values within the layer for neurons parameter: " + name);
    }

    return value;
}

template <class T>
void N2D2::Layer::setNeuronsParameter(const std::string& name, T value)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast<void (Xcell::*)(const std::string&, T)>(
                                &Xcell::setNeuronsParameter<T>),
                            std::placeholders::_1,
                            name,
                            value));
}

template <class T>
void N2D2::Layer::setNeuronsParameter(const std::string& name,
                                      T mean,
                                      Percent relStdDev)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(const std::string&, T, Percent)>(
                                &Xcell::setNeuronsParameter<T>),
                            std::placeholders::_1,
                            name,
                            mean,
                            relStdDev));
}

template <class T>
void
N2D2::Layer::setNeuronsParameter(const std::string& name, T mean, double stdDev)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(const std::string&, T, double)>(
                                &Xcell::setNeuronsParameter<T>),
                            std::placeholders::_1,
                            name,
                            mean,
                            stdDev));
}

template <class T>
void N2D2::Layer::setNeuronsParameterSpread(const std::string& name,
                                            Percent relStdDev)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(const std::string&, Percent)>(
                                &Xcell::setParameterSpread<T>),
                            std::placeholders::_1,
                            name,
                            relStdDev));
}

template <class T>
void N2D2::Layer::setNeuronsParameterSpread(const std::string& name,
                                            double stdDev)
{
    std::for_each(mCells.begin(),
                  mCells.end(),
                  std::bind(static_cast
                            <void (Xcell::*)(const std::string&, double)>(
                                &Xcell::setParameterSpread<T>),
                            std::placeholders::_1,
                            name,
                            stdDev));
}

#endif // N2D2_LAYER_H
