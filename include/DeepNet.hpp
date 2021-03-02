/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thieler@cea.fr)

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

#ifndef N2D2_DEEPNET_H
#define N2D2_DEEPNET_H

#include <string>
#include <vector>

#include "Cell/Cell.hpp"
#include "Database/Database.hpp"
#include "Xnet/Network.hpp"
#include "Target/Target.hpp"

#ifdef CUDA
#include "CudaUtils.hpp"
#include "Cell/Cell_Frame_CUDA.hpp"
#include "CMonitor_CUDA.hpp"
#endif

namespace N2D2 {

class CMonitor;
class Gnuplot;
class Monitor;


class DeepNet : public Parameterizable, public std::enable_shared_from_this<DeepNet> {
public:
    DeepNet(Network& net);

    /**
     * TODO Simplify the management of the network graph.
     * TODO Add a way to adapt the target when a cell is added after a cell with a target.
     *      Currently the target isn't moved.
     */
    void addCell(const std::shared_ptr<Cell>& cell,
                 const std::vector<std::shared_ptr<Cell>>& parents);
    void addCellAfter(const std::shared_ptr<Cell>& newCell,
                      const std::shared_ptr<Cell>& parent);
    void addCellBefore(const std::shared_ptr<Cell>& newCell,
                       const std::shared_ptr<Cell>& child);
    void addCellBetween(const std::shared_ptr<Cell>& newCell,
                        const std::shared_ptr<Cell>& parent,
                        const std::shared_ptr<Cell>& child);
    void removeCell(const std::shared_ptr<Cell>& cell, bool reconnect = true);

    /**
     * Generate a cell name that doesn't exist in the DeepNet. 
     * 
     * The 'baseName' parameter will be used as initial name and if a cell with the 
     * same name already exists in the DeepNe,t a suffix will be generated and appended
     * to the 'baseName' to avoid any collision.
     */
    std::string generateNewCellName(const std::string& baseName) const;


    void addTarget(const std::shared_ptr<Target>& cell);
    void addMonitor(const std::string& name,
                    const std::shared_ptr<Monitor>& monitor);
    void addCMonitor(const std::string& name,
                    const std::shared_ptr<CMonitor>& monitor);
                    
    std::vector<std::pair<std::string, long long int>> getCMonitorOutputsActivities();
    std::vector<std::pair<std::string, long long unsigned int>> getCMonitorFiringRates();
    std::vector<std::pair<std::string, long long unsigned int>>
    update(bool log, Time_T start, Time_T stop = 0, bool update = true);
    void save(const std::string& dirName) const;
    void load(const std::string& dirName);
    void saveNetworkParameters() const;
    void loadNetworkParameters();
    void exportNetworkFreeParameters(const std::string& dirName) const;
    void exportNetworkSolverParameters(const std::string& dirName) const;
    void importNetworkFreeParameters(const std::string& dirName,
                                     bool ignoreNotExists = false);
    void importNetworkFreeParameters(const std::string& dirName, const std::string& weightName);
    void importNetworkSolverParameters(const std::string& dirName);
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void initialize();
    void learn(std::vector<std::pair<std::string, double> >* timings = NULL);
    void test(Database::StimuliSet set = Database::Test,
              std::vector<std::pair<std::string, double> >* timings = NULL);
    void propagate(Database::StimuliSet set,
                   bool inference,
                std::vector<std::pair<std::string, double> >* timings = NULL);
    void backPropagate(std::vector<std::pair<std::string, double> >* timings = NULL);
    void update(std::vector<std::pair<std::string, double> >* timings = NULL);
    void cTicks(Time_T start, Time_T stop, Time_T timestep, bool record=false);
    void cTargetsProcess(Database::StimuliSet set = Database::Test);
    void cReset(Time_T timestamp = 0);
    void initializeCMonitors(unsigned int nbTimesteps);
    void spikeCodingCompare(const std::string& dirName, unsigned int idx) const;

    void fuseBatchNormWithConv();
    void fusePadding();
    void removeDropout();

    // Setters
    void setDatabase(const std::shared_ptr<Database>& database)
    {
        mDatabase = database;
    }
    void setStimuliProvider(const std::shared_ptr<StimuliProvider>& sp)
    {
        mStimuliProvider = sp;
    }
    template <class T>
    void setCellsParameter(const std::string& name,
                           T value,
                           bool ignoreUnknown = false);
    template <class T>
    void setCellsParameter(const std::string& name,
                           T mean,
                           Percent relStdDev,
                           bool ignoreUnknown = false);
    template <class T>
    void setCellsParameter(const std::string& name,
                           T mean,
                           double stdDev,
                           bool ignoreUnknown = false);

    // Getters
    Network& getNetwork()
    {
        return mNet;
    };
    std::shared_ptr<Database> getDatabase() const
    {
        return mDatabase;
    };
    std::shared_ptr<StimuliProvider> getStimuliProvider() const
    {
        return mStimuliProvider;
    };
    template <class T = Cell>
    std::shared_ptr<T> getCell(const std::string& name) const;
    std::map<std::string, std::shared_ptr<Cell> >& getCells()
    {
        return mCells;
    };

    bool hasCell(const std::string& name) const;


    std::shared_ptr<Monitor> getMonitor(const std::string& name) const;
    std::shared_ptr<CMonitor> getCMonitor(const std::string& name) const;
    const std::vector<std::vector<std::string> >& getLayers() const
    {
        return mLayers;
    };
    const std::vector<std::string>& getLayer(unsigned int layer) const
    {
        return mLayers.at(layer);
    };
    std::vector<std::shared_ptr<Cell> > getChildCells(const std::string
                                                       & name) const;
    std::vector<std::shared_ptr<Cell> > getParentCells(const std::string
                                                       & name) const;
    template <class T = Cell>
    std::shared_ptr<T> getTargetCell(unsigned int index = 0) const;
    template <class T = Cell>
    std::shared_ptr<T> getTargetCell(const std::string& name) const;
    const std::vector<std::shared_ptr<Target> >& getTargets() const
    {
        return mTargets;
    };
    template <class T = Target>
    std::shared_ptr<T> getTarget(unsigned int index = 0) const;
    template <class T = Target>
    std::shared_ptr<T> getTarget(const std::string& name) const;
    void getStats(Cell::Stats& stats) const;
    std::vector<unsigned int> getReceptiveField(const std::string& name,
                                const std::vector<unsigned int>& outputField
                                        = std::vector<unsigned int>()) const;

    // Clear
    void clearAll();
    void clearActivity();
    void clearFiringRate();
    void clearSuccess();
    void clear(Database::StimuliSet set);

    // Logs
    void logOutputs(const std::string& dirName,
                    unsigned int batchPos = 0) const;
    void logDiffInputs(const std::string& dirName,
                       unsigned int batchPos = 0) const;
    void logFreeParameters(const std::string& dirName) const;
    void logSchedule(const std::string& dirName) const;
    void logStats(const std::string& dirName) const;
    void logSpikeStats(const std::string& dirName,
                       unsigned int nbPatterns) const;
    void log(const std::string& baseName, Database::StimuliSet set) const;
    void logLabelsMapping(const std::string& fileName) const;
    void logEstimatedLabels(const std::string& dirName) const;
    void logEstimatedLabelsJSON(const std::string& dirName) const;
    void logLabelsLegend(const std::string& fileName) const;
    void logTimings(const std::string& fileName,
                    const std::vector
                    <std::pair<std::string, double> >& timings) const;
    void logReceptiveFields(const std::string& fileName) const;

    virtual ~DeepNet() {};

    static void drawHistogram(std::string title, const std::string& dataFileName,
                   unsigned int fileRow, unsigned int maxLabelSize, bool isLog,
                   Gnuplot& p);

protected:
    Parameter<std::string> mName;

private:
    Network& mNet;
    std::shared_ptr<Database> mDatabase;
    std::shared_ptr<StimuliProvider> mStimuliProvider;
    std::map<std::string, std::shared_ptr<Cell> > mCells;
    std::vector<std::shared_ptr<Target> > mTargets;
    std::map<std::string, std::shared_ptr<Monitor> > mMonitors;
    std::map<std::string, std::shared_ptr<CMonitor> > mCMonitors;
    std::vector<std::vector<std::string> > mLayers;

    // cellName -> parentsNames
    std::multimap<std::string, std::string> mParentLayers;
    unsigned int mStreamIdx;
    unsigned int mStreamTestIdx;
};
}

template <class T>
void N2D2::DeepNet::setCellsParameter(const std::string& name,
                                      T value,
                                      bool ignoreUnknown)
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        if ((*it).second->isParameter(name)) {
            std::cout << "Notice: setting parameter \"" << name << "\" to "
                      << value << " for cell \"" + (*it).first + "\""
                      << std::endl;
            (*it).second->setParameter(name, value);
        } else if (!ignoreUnknown)
            throw std::runtime_error("Parameter \"" + name
                                     + "\" does not exist for cell \""
                                     + (*it).first + "\"");
    }
}

template <class T>
void N2D2::DeepNet::setCellsParameter(const std::string& name,
                                      T mean,
                                      Percent relStdDev,
                                      bool ignoreUnknown)
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        if ((*it).second->isParameter(name)) {
            std::cout << "Notice: setting parameter \"" << name << "\" to "
                      << mean << " (+/- " << relStdDev << "%)"
                      << " for cell \"" + (*it).first + "\"" << std::endl;
            (*it).second->setParameter(name, mean, relStdDev);
        } else if (!ignoreUnknown)
            throw std::runtime_error("Parameter \"" + name
                                     + "\" does not exist for cell \""
                                     + (*it).first + "\"");
    }
}

template <class T>
void N2D2::DeepNet::setCellsParameter(const std::string& name,
                                      T mean,
                                      double stdDev,
                                      bool ignoreUnknown)
{
    for (std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
         = mCells.begin(),
         itEnd = mCells.end();
         it != itEnd;
         ++it) {
        if ((*it).second->isParameter(name)) {
            std::cout << "Notice: setting parameter \"" << name << "\" to "
                      << mean << " (+/- " << stdDev << ")"
                      << " for cell \"" + (*it).first + "\"" << std::endl;
            (*it).second->setParameter(name, mean, stdDev);
        } else if (!ignoreUnknown)
            throw std::runtime_error("Parameter \"" + name
                                     + "\" does not exist for cell \""
                                     + (*it).first + "\"");
    }
}

template <class T>
std::shared_ptr<T> N2D2::DeepNet::getCell(const std::string& name) const
{
    const std::map<std::string, std::shared_ptr<Cell> >::const_iterator it
        = mCells.find(name);

    if (it == mCells.end())
        throw std::runtime_error("DeepNet::getCell(): layer " + name
                                 + " does not exist");

    std::shared_ptr<T> cell = std::dynamic_pointer_cast<T>((*it).second);

    if (!cell)
        throw std::runtime_error("DeepNet::getCell(): wrong type for layer "
                                 + name);

    return cell;
}

template <class T>
std::shared_ptr<T> N2D2::DeepNet::getTargetCell(unsigned int index) const
{
    std::shared_ptr<Target> target = getTarget(index);
    std::shared_ptr<T> cell = std::dynamic_pointer_cast<T>(target->getCell());

    if (!cell) {
        std::ostringstream indexStr;
        indexStr << index;

        throw std::runtime_error(
            "DeepNet::getTargetCell(): wrong cell type for index "
            + indexStr.str());
    }

    return cell;
}

template <class T>
std::shared_ptr<T> N2D2::DeepNet::getTargetCell(const std::string& name) const
{
    std::shared_ptr<Target> target = getTarget(name);
    std::shared_ptr<T> cell = std::dynamic_pointer_cast<T>(target->getCell());

    if (!cell) {
        throw std::runtime_error(
            "DeepNet::getTargetCell(): wrong cell type for target with name: "
            + name);
    }

    return cell;
}

template <class T>
std::shared_ptr<T> N2D2::DeepNet::getTarget(unsigned int index) const
{
    std::ostringstream indexStr;
    indexStr << index;

    if (index >= mTargets.size())
        throw std::runtime_error("DeepNet::getTarget(): wrong target index: "
                                 + indexStr.str());

    std::shared_ptr<T> target = std::dynamic_pointer_cast<T>(mTargets[index]);

    if (!target)
        throw std::runtime_error(
            "DeepNet::getTarget(): wrong target type for index "
            + indexStr.str());

    return target;
}

template <class T>
std::shared_ptr<T> N2D2::DeepNet::getTarget(const std::string& name) const
{
    std::vector<std::shared_ptr<Target> >::const_iterator it
        = find_if(mTargets.begin(), mTargets.end(),
            [&name](const std::shared_ptr<Target>& target)
                {return target->getName() == name;});

    if (it == mTargets.end()) {
        throw std::runtime_error("DeepNet::getTarget(): no target with name: "
                                 + name);
    }

    std::shared_ptr<T> target = std::dynamic_pointer_cast<T>(*it);

    if (!target) {
        throw std::runtime_error(
            "DeepNet::getTarget(): wrong target type for target with name: "
            + name);
    }

    return target;
}

/** @mainpage N2D2 Index Page
 *
 * @section intro_sec Introduction
 *
 * The N2D2 sub-project allows you to create and study convolutional neural
 *network, using standard back-propagation
 * learning and/or spike-based learning or forward propagation.
 *
 * @section ini_sec Network INI File Description Syntax
 * @subsection global Global Parameters
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>DefaultModel</tt></td>
 *      <td>@p Static</td>
 *      <td>Default synaptic model for the layers. Can be any of @p Static, @p
 *Analog, @p RRAM or @p PCM</td>
 *  </tr>
 *  <tr>
 *      <td><tt>ProgramMethod(RRAM)</tt></td>
 *      <td>@p Ideal</td>
 *      <td>Programming method for RRAM synapses. Can be any of
 *<tt>N2D2::Synapse_RRAM::ProgramMethod</tt></td>
 *  </tr>
 *  <tr>
 *      <td><tt>ProgramMethod(PCM)</tt></td>
 *      <td>@p Ideal</td>
 *      <td>Programming method for PCM synapses. Can be any of
 *<tt>N2D2::Synapse_PCM::ProgramMethod</tt></td>
 *  </tr>
 *  <tr>
 *      <td><tt>CheckWeightRange</tt></td>
 *      <td>1</td>
 *      <td>Check synaptic weight range when loading the weights</td>
 *  </tr>
 *  <tr>
 *      <td><tt>LearningRateDecay</tt></td>
 *      <td>1.0</td>
 *      <td>Learning rate decay during static learning. After each epoch, the
 *learning rate is multiplied by @p
 *          LearningRateDecay</td>
 *  </tr>
 *  </table>
 *
 * @subsection env Environment Layer
 *
 * The environment layer defines the input of the network, and must be defined
 *in the <b><tt>[env]</tt></b> mandatory section.
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>SizeX</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Base width (at scale 1.0) of the environment</td>
 *  </tr>
 *  <tr>
 *      <td><tt>SizeY</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Base height (at scale 1.0) of the environment</td>
 *  </tr>
 *  <tr>
 *      <td><tt>ConfigSection</tt></td>
 *      <td></td>
 *      <td>Name of the configuration section for
 *<tt>N2D2::Environment</tt></td>
 *  </tr>
 *  </table>
 *
 * @subsubsection filter Environment Filters Definition
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>Scale</tt></td>
 *      <td>1.0</td>
 *      <td>Scale of the filter (> 0.0 and <= 1.0)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Kernel</tt></td>
 *      <td></td>
 *      <td>Kernel type of the filter</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Channel</tt></td>
 *      <td>@p Gray</td>
 *      <td>Channel of the filter. Can be any of
 *<tt>N2D2::Filter::Channel</tt></td>
 *  </tr>
 *  </table>
 *
 * @subsection layer Layers Definition
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>Input</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Name of the section(s) for the input layer(s). Comma separated</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Type</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Type of the layer. Can be any of @p Conv, @p Lc, @p Pool, @p Fc or
 *@p Rbf</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Model</tt></td>
 *      <td>@p DefaultModel value</td>
 *      <td>Synaptic model for this layer. Can be any of @p Static, @p Analog,
 *@p RRAM or @p PCM</td>
 *  </tr>
 *  <tr>
 *      <td><tt>ConfigSection</tt></td>
 *      <td></td>
 *      <td>Name of the configuration section for layer</td>
 *  </tr>
 *  </table>
 *
 * @subsubsection ConvCell ConvCell Layer
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>KernelWidth</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Width of the kernel</td>
 *  </tr>
 *  <tr>
 *      <td><tt>KernelHeight</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Height of the kernel</td>
 *  </tr>
 *  <tr>
 *      <td><tt>NbOutputs</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Number of output channels</td>
 *  </tr>
 *  <tr>
 *      <td><tt>SubSampleX</tt></td>
 *      <td>1</td>
 *      <td>X-axis subsampling factor of the output feature maps</td>
 *  </tr>
 *  <tr>
 *      <td><tt>SubSampleY</tt></td>
 *      <td>1</td>
 *      <td>Y-axis subsampling factor of the output feature maps</td>
 *  </tr>
 *  <tr>
 *      <td><tt>SubSample</tt></td>
 *      <td></td>
 *      <td>Subsampling factor of the output feature maps (mutually exclusive
 *with @p SubSampleX and @p SubSampleY)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>StrideX</tt></td>
 *      <td>1</td>
 *      <td>X-axis stride of the kernels</td>
 *  </tr>
 *  <tr>
 *      <td><tt>StrideY</tt></td>
 *      <td>1</td>
 *      <td>Y-axis stride of the kernels</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Stride</tt></td>
 *      <td></td>
 *      <td>Stride of the kernels (mutually exclusive with @p StrideX and @p
 *StrideY)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>PaddingX</tt></td>
 *      <td>0</td>
 *      <td>X-axis input padding</td>
 *  </tr>
 *  <tr>
 *      <td><tt>PaddingY</tt></td>
 *      <td>0</td>
 *      <td>Y-axis input padding</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Padding</tt></td>
 *      <td></td>
 *      <td>Input padding (mutually exclusive with @p PaddingX and @p
 *PaddingY)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>ActivationFunction</tt></td>
 *      <td>@p Tanh</td>
 *      <td>Activation function. Can be any of @p Sigmoid, @p Rectifier, @p
 *Linear or @p Tanh</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.SizeX</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas pattern default width</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.SizeY</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas pattern default height</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Size</tt></td>
 *      <td></td>
 *      <td>Mapping canvas pattern default size (mutually exclusive with
 *<tt>Mapping.SizeX</tt> and <tt>Mapping.SizeY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.StrideX</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas default X-axis step</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.StrideY</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas default Y-axis step</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Stride</tt></td>
 *      <td></td>
 *      <td>Mapping canvas default step (mutually exclusive with
 *<tt>Mapping.StrideX</tt> and <tt>Mapping.StrideY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.OffsetX</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas default X-axis offset</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.OffsetY</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas default Y-axis offset</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Offset</tt></td>
 *      <td></td>
 *      <td>Mapping canvas default offset (mutually exclusive with
 *<tt>Mapping.OffsetX</tt> and <tt>Mapping.OffsetY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.NbIterations</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas pattern default number of iterations (0 means no
 *limit)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).SizeX</tt></td>
 *      <td><tt>Mapping.SizeX</tt> value</td>
 *      <td>Mapping canvas pattern width for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).SizeY</tt></td>
 *      <td><tt>Mapping.SizeY</tt> value</td>
 *      <td>Mapping canvas pattern height for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Size</tt></td>
 *      <td></td>
 *      <td>Mapping canvas pattern default size for input layer @a in (mutually
 *exclusive with <tt>Mapping(@a in).SizeX</tt> and
 *          <tt>Mapping(@a in).SizeY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).StrideX</tt></td>
 *      <td><tt>Mapping.StrideX</tt> value</td>
 *      <td>Mapping canvas X-axis step for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).StrideY</tt></td>
 *      <td><tt>Mapping.StrideY</tt> value</td>
 *      <td>Mapping canvas Y-axis step for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Stride</tt></td>
 *      <td></td>
 *      <td>Mapping canvas step for input layer @a in (mutually exclusive with
 *<tt>Mapping(@a in).StrideX</tt> and
 *          <tt>Mapping(@a in).StrideY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).OffsetX</tt></td>
 *      <td><tt>Mapping.OffsetX</tt> value</td>
 *      <td>Mapping canvas X-axis offset for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).OffsetY</tt></td>
 *      <td><tt>Mapping.OffsetY</tt> value</td>
 *      <td>Mapping canvas Y-axis offset for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Offset</tt></td>
 *      <td></td>
 *      <td>Mapping canvas offset for input layer @a in (mutually exclusive with
 *<tt>Mapping(@a in).OffsetX</tt> and
 *          <tt>Mapping(@a in).OffsetY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).NbIterations</tt></td>
 *      <td><tt>Mapping.NbIterations</tt> value</td>
 *      <td>Mapping canvas pattern number of iterations for input layer @a in (0
 *means no limit)</td>
 *  </tr>
 *  </table>
 *
 * @subsubsection LcCell LcCell Layer
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>KernelWidth</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Width of the kernel</td>
 *  </tr>
 *  <tr>
 *      <td><tt>KernelHeight</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Height of the kernel</td>
 *  </tr>
 *  <tr>
 *      <td><tt>NbOutputs</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Number of output channels</td>
 *  </tr>
 *  <tr>
 *      <td><tt>SubSampleX</tt></td>
 *      <td>1</td>
 *      <td>X-axis subsampling factor of the output feature maps</td>
 *  </tr>
 *  <tr>
 *      <td><tt>SubSampleY</tt></td>
 *      <td>1</td>
 *      <td>Y-axis subsampling factor of the output feature maps</td>
 *  </tr>
 *  <tr>
 *      <td><tt>SubSample</tt></td>
 *      <td></td>
 *      <td>Subsampling factor of the output feature maps (mutually exclusive
 *with @p SubSampleX and @p SubSampleY)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>StrideX</tt></td>
 *      <td>1</td>
 *      <td>X-axis stride of the kernels</td>
 *  </tr>
 *  <tr>
 *      <td><tt>StrideY</tt></td>
 *      <td>1</td>
 *      <td>Y-axis stride of the kernels</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Stride</tt></td>
 *      <td></td>
 *      <td>Stride of the kernels (mutually exclusive with @p StrideX and @p
 *StrideY)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>PaddingX</tt></td>
 *      <td>0</td>
 *      <td>X-axis input padding</td>
 *  </tr>
 *  <tr>
 *      <td><tt>PaddingY</tt></td>
 *      <td>0</td>
 *      <td>Y-axis input padding</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Padding</tt></td>
 *      <td></td>
 *      <td>Input padding (mutually exclusive with @p PaddingX and @p
 *PaddingY)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>ActivationFunction</tt></td>
 *      <td>@p Tanh</td>
 *      <td>Activation function. Can be any of @p Sigmoid, @p Rectifier, @p
 *Linear or @p Tanh</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.SizeX</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas pattern default width</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.SizeY</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas pattern default height</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Size</tt></td>
 *      <td></td>
 *      <td>Mapping canvas pattern default size (mutually exclusive with
 *<tt>Mapping.SizeX</tt> and <tt>Mapping.SizeY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.StrideX</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas default X-axis step</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.StrideY</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas default Y-axis step</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Stride</tt></td>
 *      <td></td>
 *      <td>Mapping canvas default step (mutually exclusive with
 *<tt>Mapping.StrideX</tt> and <tt>Mapping.StrideY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.OffsetX</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas default X-axis offset</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.OffsetY</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas default Y-axis offset</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Offset</tt></td>
 *      <td></td>
 *      <td>Mapping canvas default offset (mutually exclusive with
 *<tt>Mapping.OffsetX</tt> and <tt>Mapping.OffsetY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.NbIterations</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas pattern default number of iterations (0 means no
 *limit)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).SizeX</tt></td>
 *      <td><tt>Mapping.SizeX</tt> value</td>
 *      <td>Mapping canvas pattern width for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).SizeY</tt></td>
 *      <td><tt>Mapping.SizeY</tt> value</td>
 *      <td>Mapping canvas pattern height for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Size</tt></td>
 *      <td></td>
 *      <td>Mapping canvas pattern default size for input layer @a in (mutually
 *exclusive with <tt>Mapping(@a in).SizeX</tt> and
 *          <tt>Mapping(@a in).SizeY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).StrideX</tt></td>
 *      <td><tt>Mapping.StrideX</tt> value</td>
 *      <td>Mapping canvas X-axis step for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).StrideY</tt></td>
 *      <td><tt>Mapping.StrideY</tt> value</td>
 *      <td>Mapping canvas Y-axis step for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Stride</tt></td>
 *      <td></td>
 *      <td>Mapping canvas step for input layer @a in (mutually exclusive with
 *<tt>Mapping(@a in).StrideX</tt> and
 *          <tt>Mapping(@a in).StrideY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).OffsetX</tt></td>
 *      <td><tt>Mapping.OffsetX</tt> value</td>
 *      <td>Mapping canvas X-axis offset for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).OffsetY</tt></td>
 *      <td><tt>Mapping.OffsetY</tt> value</td>
 *      <td>Mapping canvas Y-axis offset for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Offset</tt></td>
 *      <td></td>
 *      <td>Mapping canvas offset for input layer @a in (mutually exclusive with
 *<tt>Mapping(@a in).OffsetX</tt> and
 *          <tt>Mapping(@a in).OffsetY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).NbIterations</tt></td>
 *      <td><tt>Mapping.NbIterations</tt> value</td>
 *      <td>Mapping canvas pattern number of iterations for input layer @a in (0
 *means no limit)</td>
 *  </tr>
 *  </table>
 *
 * @subsubsection PoolCell PoolCell Layer
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>PoolWidth</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Width of the pooling area</td>
 *  </tr>
 *  <tr>
 *      <td><tt>PoolHeight</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Height of the pooling area</td>
 *  </tr>
 *  <tr>
 *      <td><tt>NbOutputs</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Number of output channels</td>
 *  </tr>
 *  <tr>
 *      <td><tt>StrideX</tt></td>
 *      <td>1</td>
 *      <td>X-axis stride of the pooling areas</td>
 *  </tr>
 *  <tr>
 *      <td><tt>StrideY</tt></td>
 *      <td>1</td>
 *      <td>Y-axis stride of the pooling areas</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Stride</tt></td>
 *      <td></td>
 *      <td>Stride of the pooling areas (mutually exclusive with @p StrideX and
 *@p StrideY)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>ActivationFunction</tt></td>
 *      <td>@p Linear</td>
 *      <td>Activation function. Can be any of @p Sigmoid, @p Rectifier, @p
 *Linear or @p Tanh</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.SizeX</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas pattern default width</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.SizeY</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas pattern default height</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Size</tt></td>
 *      <td></td>
 *      <td>Mapping canvas pattern default size (mutually exclusive with
 *<tt>Mapping.SizeX</tt> and <tt>Mapping.SizeY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.StrideX</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas default X-axis step</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.StrideY</tt></td>
 *      <td>1</td>
 *      <td>Mapping canvas default Y-axis step</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Stride</tt></td>
 *      <td></td>
 *      <td>Mapping canvas default step (mutually exclusive with
 *<tt>Mapping.StrideX</tt> and <tt>Mapping.StrideY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.OffsetX</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas default X-axis offset</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.OffsetY</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas default Y-axis offset</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.Offset</tt></td>
 *      <td></td>
 *      <td>Mapping canvas default offset (mutually exclusive with
 *<tt>Mapping.OffsetX</tt> and <tt>Mapping.OffsetY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping.NbIterations</tt></td>
 *      <td>0</td>
 *      <td>Mapping canvas pattern default number of iterations (0 means no
 *limit)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).SizeX</tt></td>
 *      <td><tt>Mapping.SizeX</tt> value</td>
 *      <td>Mapping canvas pattern width for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).SizeY</tt></td>
 *      <td><tt>Mapping.SizeY</tt> value</td>
 *      <td>Mapping canvas pattern height for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Size</tt></td>
 *      <td></td>
 *      <td>Mapping canvas pattern default size for input layer @a in (mutually
 *exclusive with <tt>Mapping(@a in).SizeX</tt> and
 *          <tt>Mapping(@a in).SizeY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).StrideX</tt></td>
 *      <td><tt>Mapping.StrideX</tt> value</td>
 *      <td>Mapping canvas X-axis step for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).StrideY</tt></td>
 *      <td><tt>Mapping.StrideY</tt> value</td>
 *      <td>Mapping canvas Y-axis step for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Stride</tt></td>
 *      <td></td>
 *      <td>Mapping canvas step for input layer @a in (mutually exclusive with
 *<tt>Mapping(@a in).StrideX</tt> and
 *          <tt>Mapping(@a in).StrideY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).OffsetX</tt></td>
 *      <td><tt>Mapping.OffsetX</tt> value</td>
 *      <td>Mapping canvas X-axis offset for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).OffsetY</tt></td>
 *      <td><tt>Mapping.OffsetY</tt> value</td>
 *      <td>Mapping canvas Y-axis offset for input layer @a in</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).Offset</tt></td>
 *      <td></td>
 *      <td>Mapping canvas offset for input layer @a in (mutually exclusive with
 *<tt>Mapping(@a in).OffsetX</tt> and
 *          <tt>Mapping(@a in).OffsetY</tt>)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>Mapping(@a in).NbIterations</tt></td>
 *      <td><tt>Mapping.NbIterations</tt> value</td>
 *      <td>Mapping canvas pattern number of iterations for input layer @a in (0
 *means no limit)</td>
 *  </tr>
 *  </table>
 *
 * @subsubsection FcCell FcCell Layer
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>NbOutputs</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Number of output neurons</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputOffsetX</tt></td>
 *      <td>0</td>
 *      <td>X-axis offset for input layer connection</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputOffsetY</tt></td>
 *      <td>0</td>
 *      <td>Y-axis offset for input layer connection (non applicable for 1D
 *inputs)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputWidth</tt></td>
 *      <td>0</td>
 *      <td>Width of the input layer connection, if 0, use the full input
 *width</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputHeight</tt></td>
 *      <td>0</td>
 *      <td>Height of the input layer connection, if 0, use the full input
 *height (non applicable for 1D inputs)</td>
 *  </tr>
 *  </table>
 *
 * @subsubsection RbfCell RbfCell Layer
 *
 *  <table>
 *  <tr>
 *      <th>Parameter</th>
 *      <th>Default value</th>
 *      <th>Description</th>
 *  </tr>
 *  <tr>
 *      <td><tt>NbOutputs</tt></td>
 *      <td>[REQUIRED]</td>
 *      <td>Number of output neurons</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputOffsetX</tt></td>
 *      <td>0</td>
 *      <td>X-axis offset for input layer connection</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputOffsetY</tt></td>
 *      <td>0</td>
 *      <td>Y-axis offset for input layer connection (non applicable for 1D
 *inputs)</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputWidth</tt></td>
 *      <td>0</td>
 *      <td>Width of the input layer connection, if 0, use the full input
 *width</td>
 *  </tr>
 *  <tr>
 *      <td><tt>InputHeight</tt></td>
 *      <td>0</td>
 *      <td>Height of the input layer connection, if 0, use the full input
 *height (non applicable for 1D inputs)</td>
 *  </tr>
 *  </table>
*/

#endif // N2D2_DEEPNET_H
