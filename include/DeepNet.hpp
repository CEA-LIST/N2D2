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
#include <atomic>

#include "Cell/Cell.hpp"
#include "Database/Database.hpp"
#include "Xnet/Network.hpp"
#include "Target/Target.hpp"

#ifdef CUDA
#include "CudaUtils.hpp"
#include "Cell/Cell_Frame_CUDA.hpp"
#include "CMonitor_CUDA.hpp"

#ifdef NVML
#include <nvml.h>
#endif

#endif

namespace N2D2 {

class CMonitor;
class Gnuplot;
class Monitor;


class DeepNet : public Parameterizable, public std::enable_shared_from_this<DeepNet> {
public:
    DeepNet(Network& net);

    typedef struct {
        std::vector<char> finished;
        std::atomic<bool> firstFinished;
        std::atomic<unsigned int> nbFinished;
        std::atomic<unsigned int> power;
    } SharedValues;

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
    void learn_singleDevice(std::vector<std::pair<std::string, double> >* timings = NULL);
#ifdef CUDA
    void learn_multiDevices(std::vector<std::pair<std::string, double> >* timings = NULL);
#endif
    void test(Database::StimuliSet set = Database::Test,
              std::vector<std::pair<std::string, double> >* timings = NULL);
    void propagate(bool inference);
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
    void insertBatchNormAfterConv(bool moveActivation = true);
    void fusePadding();
    void removeDropout();

#ifdef CUDA
    void lastBatch() {
        mLastPass = true;
    };
#endif

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
#ifdef CUDA
    void setBanAllowed(bool option)
    {
        mBanAllowed = option;
    };
    char isDeviceDropped(int dev) const
    {
        return mDropDevices[dev];
    };
#endif

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

    std::string getName() const {
        return mName;
    };

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

#ifdef CUDA
    std::vector<N2D2::DeviceState> getStates() 
    {
        return mStates;
    };
    SharedValues& getMultiDevicesInfo()
    {
        return mMultiDevicesInfo;
    };
#endif

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

    virtual ~DeepNet();

    static void drawHistogram(std::string title, const std::string& dataFileName,
                   unsigned int fileRow, unsigned int maxLabelSize, bool isLog,
                   Gnuplot& p);

protected:
    Parameter<std::string> mName;

private:
    std::string getCellModelType(const Cell& cell);

    Network& mNet;
    std::shared_ptr<Database> mDatabase;
    std::shared_ptr<StimuliProvider> mStimuliProvider;
    std::map<std::string, std::shared_ptr<Cell> > mCells;
    std::vector<std::shared_ptr<Target> > mTargets;
    std::map<std::string, std::shared_ptr<Monitor> > mMonitors;
    std::map<std::string, std::shared_ptr<CMonitor> > mCMonitors;
    std::vector<std::vector<std::string> > mLayers;

#ifdef CUDA
    /// Device states
    std::vector<N2D2::DeviceState> mStates;
    /// Vector of warnings for all devices
    std::vector<unsigned int> mDevicesWarning;
    /// Parameter to know if the batchs used during 
    /// learning are the last
    bool mLastPass;
    /// Parameter to allow banishment
    bool mBanAllowed;
    /// Number of passages allowed before banning
    unsigned int mNbPassBeforeBan;
    /// Average power usage from all connected devices
    unsigned int mAveragePowerUsage;
    /// Vector of all devices which have been dropped
    std::vector<char> mDropDevices;
    /// Information shared among all connected devices
    SharedValues mMultiDevicesInfo;
    /// Identifier of the device where data is 
    /// gathered during update
    int mMasterDevice;    
#endif

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

#endif // N2D2_DEEPNET_H
