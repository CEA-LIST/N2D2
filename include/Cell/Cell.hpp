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

#include "containers/Tensor.hpp"
#include "utils/Parameterizable.hpp"
#include "FloatT.hpp"

namespace N2D2 {

class DeepNet;
class HeteroStimuliProvider;
class StimuliProvider;

typedef unsigned int CellId_T;

/**
 * Cell is the base object for any kind of layer composing a deep network.
 * It provides the base interface required.
*/
class Cell : public Parameterizable, public std::enable_shared_from_this<Cell> {
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

    /// Free parameters type in respect to their contribution to the output:
    /// Multiplicative = weights, Additive = biases
    enum FreeParametersType {
        Additive,
        Multiplicative,
        All
    };

    /**
     * Abstract cell constructor
     *
     * @param name          Name of the cell
     * @param type          Type of the cell
     * @param nbOutputs     Number of outputs maps of the cell (if 1D = number
     *of outputs)
    */
    Cell(const DeepNet& deepNet, const std::string& name, unsigned int nbOutputs);

    bool isQuantized() const {
        return mQuantizedNbBits > 0;
    }

    void setQuantized(std::size_t nbBits) {
        mQuantizedNbBits = nbBits;
    }

    std::size_t getQuantizedNbBits() const {
        return mQuantizedNbBits;
    }


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
    size_t getNbChannels() const
    {
        return mInputsDims.back();
    };
    /// Returns cell input channels width
    size_t getChannelsWidth() const
    {
        return mInputsDims[0];
    };
    /// Returns cell input channels height
    size_t getChannelsHeight() const
    {
        return mInputsDims[1];
    };
    size_t getInputsDim(unsigned int dim) const
    {
        assert(dim < mInputsDims.size());
        return mInputsDims[dim];
    };
    const std::vector<size_t>& getInputsDims() const
    {
        return mInputsDims;
    };
    size_t getInputsSize() const
    {
        return (!mInputsDims.empty())
            ? std::accumulate(mInputsDims.begin(), mInputsDims.end(),
                              1U, std::multiplies<size_t>())
            : 0;
    };
    /// Returns number of output maps in the cell (or number of outputs for 1D
    /// cells)
    size_t getNbOutputs() const
    {
        return mOutputsDims.back();
    };
    /// Returns cell output maps width (returns 1 for 1D cells)
    size_t getOutputsWidth() const
    {
        return mOutputsDims[0];
    };
    /// Returns cell output maps height (returns 1 for 1D cells)
    size_t getOutputsHeight() const
    {
        return mOutputsDims[1];
    };
    size_t getOutputsDim(unsigned int dim) const
    {
        assert(dim < mOutputsDims.size());
        return mOutputsDims[dim];
    };
    const std::vector<size_t>& getOutputsDims() const
    {
        return mOutputsDims;
    };
    size_t getOutputsSize() const
    {
        return (!mOutputsDims.empty())
            ? std::accumulate(mOutputsDims.begin(), mOutputsDims.end(),
                              1U, std::multiplies<size_t>())
            : 0;
    };
    /// Fill cell stats
    virtual void getStats(Stats& stats) const = 0;
    /// Get cells input receptive field for a given output area
    virtual std::vector<unsigned int> getReceptiveField(
                                const std::vector<unsigned int>& /*outputField*/
                                        = std::vector<unsigned int>()) const
    {
        // return empty vector if not implemented
        return std::vector<unsigned int>();
    }

    const DeepNet& getAssociatedDeepNet() const {
        return mDeepNet;
    }

    std::vector<std::shared_ptr<Cell>> getChildrenCells() const;
    std::vector<std::shared_ptr<Cell>> getParentsCells() const;

    /**
     * Check if an output map is connected to an input channel
     *
     * @param channel       Input channel number
     * @param output        Output map number
     * @return true if the output map if connected to input channel
    */
    bool isConnection(unsigned int channel, unsigned int output) const
    {
        return mMapping(output, channel);
    };

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
                          const Tensor<bool>& mapping = Tensor<bool>()) = 0;

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
                          const Tensor<bool>& mapping = Tensor<bool>()) = 0;

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
                            const Tensor<bool>& mapping = Tensor<bool>());

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
                          const Tensor<bool>& mapping = Tensor<bool>()) = 0;

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
    
    virtual void clearInputs() = 0;

    /// Initialize the state of the cell (e.g. weights random initialization)
    virtual void initialize() {};

    virtual void initializeParameters(unsigned int /*inputDimZ*/, unsigned int /*nbInputs*/) 
    {
        throw std::runtime_error("Error: initializeParameters not implemented for this cell type!");
    };
    virtual void initializeDataDependent() 
    {
        throw std::runtime_error("Error: initializeDataDependent not implemented for this cell type!");
    };

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
     * Export Activation parameters to a file, in ASCII format compatible between
     *the different cell models
     *
     * @param fileName      Destination file
    */
    virtual void exportActivationParameters(const std::string& /*fileName*/) const {};
    virtual void exportQuantFreeParameters(const std::string& /*fileName*/) const {};

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
    virtual void importActivationParameters(const std::string& /*fileName*/,
                                            bool /*ignoreNotExists*/ = false) {};

    virtual void logFreeParameters(const std::string & /*fileName*/) const {};

    /**
     * Log cell free parameters distribution
     *
     * @param fileName      Destination file
    */
    virtual void logFreeParametersDistrib(
        const std::string& /*fileName*/,
        FreeParametersType /*type*/ = All) const {};

    /**
     * Log cell free parameters quantized distribution
     *
     * @param fileName      Destination file
    */
    virtual void logQuantFreeParametersDistrib(
        const std::string& /*fileName*/,
        FreeParametersType /*type*/ = All) const {};

    virtual std::pair<Float_T, Float_T> getFreeParametersRange(FreeParametersType /*type*/ = All) const {
        return std::pair<Float_T, Float_T>();
    };

    virtual std::pair<Float_T, Float_T> getFreeParametersRangePerOutput(
            std::size_t /*output*/, 
            FreeParametersType /*type*/ = All) const 
    {
        return std::pair<Float_T, Float_T>();
    };
    virtual std::pair<Float_T, Float_T> getFreeParametersRangePerChannel(
            std::size_t /*channel*/) const 
    {
        return std::pair<Float_T, Float_T>();
    };

    virtual void processFreeParameters(std::function<Float_T(Float_T)> /*func*/,
                                       FreeParametersType /*type*/ = All) {};
    virtual void processFreeParametersPerOutput(std::function<Float_T(Float_T)> /*func*/,
                                                std::size_t /*output*/,
                                                FreeParametersType /*type*/ = All) {};
    virtual void processFreeParametersPerChannel(std::function<Float_T(Float_T)> /*func*/,
                                                std::size_t /*channel*/) {};
    bool isFullMap() const {
        return (groupMap() == 1);
    }
    size_t groupMap() const;
    bool isUnitMap() const {
        return (mMapping.dimX() == mMapping.dimY()
                && groupMap() == mMapping.dimX());
    };
    /// Destructor
    virtual ~Cell() {};

protected:
    void setInputsDims(std::initializer_list<size_t> dims);
    virtual void setInputsDims(const std::vector<size_t>& dims);
    virtual void setOutputsDims() = 0;
    size_t getNbGroups(const Tensor<bool>& map) const;

    std::pair<double, double> getOutputsRangeParents() const;

protected:
    const CellId_T mId;
    
    // DeepNet to which the cell belongs
    const DeepNet& mDeepNet;
    const std::string mName;

    // Input dims
    std::vector<size_t> mInputsDims;
    // Output dims
    std::vector<size_t> mOutputsDims;
    // Input-output mapping
    Tensor<bool> mMapping;

    // 0 if the cell is not quantized, otherwise the number of bits used by the quantization.
    std::size_t mQuantizedNbBits;

private:
    static unsigned int mIdCnt;

    mutable size_t mGroupMap;
    mutable bool mGroupMapInitialized;
};
}

#endif // N2D2_CELL_H
