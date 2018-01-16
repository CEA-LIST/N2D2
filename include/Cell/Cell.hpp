/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CELL_H
#define N2D2_CELL_H

#include "Environment.hpp"
#include "HeteroEnvironment.hpp"
#include "utils/Parameterizable.hpp"

namespace N2D2 {
typedef unsigned int CellId_T;

/**
 * Cell is the base object for any kind of layer composing a deep network.
 * It provides the base interface required.
*/
class Cell : public Parameterizable {
public:
    struct Stats {
        Stats()
            : nbNeurons(0),
              nbNodes(0),
              nbSynapses(0),
              nbVirtualSynapses(0),
              nbConnections(0) {};

        unsigned int nbNeurons;
        unsigned int nbNodes;
        unsigned long long int nbSynapses;
        unsigned long long int nbVirtualSynapses;
        unsigned long long int nbConnections;
    };

    /**
     * Abstract cell constructor
     *
     * @param name          Name of the cell
     * @param type          Type of the cell
     * @param nbOutputs     Number of outputs maps of the cell (if 1D = number
     *of outputs)
    */
    Cell(const std::string& name, unsigned int nbOutputs);
    /// Returns cell unique ID
    CellId_T getId() const
    {
        return mId;
    };
    /// Returns cell name
    const std::string& getName() const
    {
        return mName;
    };
    /// Returns cell type
    virtual const char* getType() const = 0;
    /// Returns number of channels in the cell
    virtual unsigned int getNbChannels() const = 0;
    /// Returns cell input channels width
    unsigned int getChannelsWidth() const
    {
        return mChannelsWidth;
    };
    /// Returns cell input channels height
    unsigned int getChannelsHeight() const
    {
        return mChannelsHeight;
    };
    /// Returns number of output maps in the cell (or number of outputs for 1D
    /// cells)
    unsigned int getNbOutputs() const
    {
        return mNbOutputs;
    };
    /// Returns cell output maps width (returns 1 for 1D cells)
    unsigned int getOutputsWidth() const
    {
        return mOutputsWidth;
    };
    /// Returns cell output maps height (returns 1 for 1D cells)
    unsigned int getOutputsHeight() const
    {
        return mOutputsHeight;
    };
    /// Fill cell stats
    virtual void getStats(Stats& stats) const = 0;

    /**
     * Check if an output map is connected to an input channel
     *
     * @param channel       Input channel number
     * @param output        Output map number
     * @return true if the output map if connected to input channel
    */
    virtual bool isConnection(unsigned int channel,
                              unsigned int output) const = 0;

    /**
     * Connect an input filter from the environment to the cell
     *
     * @param sp            N2D2 StimuliProvider object reference
     * @param channel       Channel number in the environment
     * @param x0            Left offset
     * @param y0            Top offset
     * @param width         Width
     * @param height        Height
     * @param mapping       Connection between the environment map and the cell
     *output maps (size of the vector = number of
     * output maps in the cell)
    */
    virtual void addInput(StimuliProvider& sp,
                          unsigned int channel,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width,
                          unsigned int height,
                          const std::vector<bool>& mapping = std::vector
                          <bool>()) = 0;

    /**
     * Connect an input map from the environment to the cell
     *
     * @param sp            N2D2 StimuliProvider object reference
     * @param x0            Left offset
     * @param y0            Top offset
     * @param width         Width
     * @param height        Height
     * @param mapping       Connection between the environment map filters and
     *the cell output maps (size of the matrix = number
     * of output maps in the cell [cols] x number of filters in the environment
     *map [rows])
    */
    virtual void addInput(StimuliProvider& sp,
                          unsigned int x0 = 0,
                          unsigned int y0 = 0,
                          unsigned int width = 0,
                          unsigned int height = 0,
                          const Matrix<bool>& mapping = Matrix<bool>()) = 0;

    /**
     * Connect all the input maps from the environment to the cell
     *
     * @param sp            N2D2 StimuliProvider object reference
     * @param x0            Left offset
     * @param y0            Top offset
     * @param width         Width
     * @param height        Height
     * @param mapping       Connection between the environment map filters and
     *the cell output maps (size of the matrix = number
     * of output maps in the cell [cols] x total number of filters in the
     *environment [rows])
    */
    void addMultiscaleInput(HeteroStimuliProvider& sp,
                            unsigned int x0 = 0,
                            unsigned int y0 = 0,
                            unsigned int width = 0,
                            unsigned int height = 0,
                            const Matrix<bool>& mapping = Matrix<bool>());

    /**
     * Connect an input cell to the cell
     *
     * @param cell          Pointer to the input cell
     * @param mapping       Connection between the input cell output maps (input
     *channels) and the cell output maps (size of the
     * matrix = number of output maps in the cell [cols] x number of input cell
     *output maps (input channels) [rows])
    */
    virtual void addInput(Cell* cell,
                          const Matrix<bool>& mapping = Matrix<bool>()) = 0;

    /**
     * Connect an input cell to the cell
     *
     * @param cell          Pointer to the input cell
     * @param x0            Left offset
     * @param y0            Top offset
     * @param width         Width
     * @param height        Height
    */
    virtual void addInput(Cell* cell,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width = 0,
                          unsigned int height = 0) = 0;

    /// Initialize the state of the cell (e.g. weights random initialization)
    virtual void initialize() {};

    /**
     * Save cell configuration and free parameters to a directory
     *
     * @param dirName       Destination directory
    */
    virtual void save(const std::string& dirName) const;

    /**
     * Load cell configuration and free parameters from a directory
     *
     * @param dirName       Source directory
    */
    virtual void load(const std::string& dirName);

    /**
     * Save cell free parameters to a file
     *
     * @param fileName      Destination file
    */
    virtual void saveFreeParameters(const std::string& /*fileName*/) const {};

    /**
     * Load cell free parameters from a file
     *
     * @param fileName      Source file
     * @param ignoreNotExists If true, don't throw an error if the file doesn't
     *exist
    */
    virtual void loadFreeParameters(const std::string& /*fileName*/,
                                    bool /*ignoreNotExists*/ = false) {};

    /**
     * Export cell free parameters to a file, in ASCII format compatible between
     *the different cell models
     *
     * @param fileName      Destination file
    */
    virtual void exportFreeParameters(const std::string& /*fileName*/) const {};

    /**
     * Load cell free parameters from a file, in ASCII format compatible between
     *the different cell models
     *
     * @param fileName      Source file
     * @param ignoreNotExists If true, don't throw an error if the file doesn't
     *exist
    */
    virtual void importFreeParameters(const std::string& /*fileName*/,
                                      bool /*ignoreNotExists*/ = false) {};
    virtual void logFreeParameters(const std::string & /*fileName*/) const {};

    /**
     * Log cell free parameters distribution
     *
     * @param fileName      Destination file
    */
    virtual void logFreeParametersDistrib(const std::string
                                          & /*fileName*/) const {};

    /**
     * Discretize cell free parameters
     *
     * @param nbLevels      Number of discrete levels
    */
    virtual void discretizeFreeParameters(unsigned int /*nbLevels*/) {};
    virtual std::pair<Float_T, Float_T> getFreeParametersRange() const {
        return std::pair<Float_T, Float_T>();
    };
    virtual void processFreeParameters(const std::function
                                       <double(const double&)>& /*func*/) {};

    /// Destructor
    virtual ~Cell() {};

protected:
    virtual void setInputsSize(unsigned int width, unsigned int height);
    virtual void setOutputsSize() = 0;

    const CellId_T mId;
    const std::string mName;

    // Input width
    unsigned int mChannelsWidth;
    // Input height
    unsigned int mChannelsHeight;
    // Number of output feature maps
    const unsigned int mNbOutputs;
    // Input width
    unsigned int mOutputsWidth;
    // Input height
    unsigned int mOutputsHeight;

private:
    static unsigned int mIdCnt;
};
}

#endif // N2D2_CELL_H
