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

#ifndef N2D2_MONITOR_H
#define N2D2_MONITOR_H

#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "Layer.hpp"
#include "Network.hpp"
#include "NodeNeuron.hpp"
#include "Xcell.hpp"
#include "utils/Gnuplot.hpp"

namespace N2D2 {
/**
 * The Monitor class provides tools to monitor the activity of the network.
 * For example, it provides methods to automatically compute the recognition
 * rate of an unsupervised learning network.
 * Its aim is also to facilitate visual representation of the network's state
 * and its evolution.
*/
class Monitor {
public:
    Monitor(Network& net);
    void add(Node& node);
    template <class T> void add(const std::vector<T*>& nodes);
    void add(Xcell& cell);
    void add(Layer& layer);
    void recordEvent(EventType_T type)
    {
        mRecordEventTypes.insert(type);
    };
    void recordAllEvents()
    {
        mRecordEventTypes.clear();
    };
    virtual void update(bool recordActivity = false);
    bool checkLearning(unsigned int cls,
                       NodeId_T targetId,
                       bool winnerIsEarlier = false,
                       bool update = true);
    bool checkLearningResponse(unsigned int cls,
                               NodeId_T targetId,
                               NodeId_T responseId,
                               bool update = true);
    bool checkLearning(unsigned int cls,
                       const std::vector<NodeId_T>& targetIds,
                       bool winnerIsEarlier = false,
                       bool update = true);
    bool checkLearningResponse(unsigned int cls,
                               const std::vector<NodeId_T>& targetIds,
                               NodeId_T responseId,
                               bool update = true);
    bool checkLearning(unsigned int cls,
                       bool winnerIsEarlier = false,
                       bool update = true);
    bool checkLearningResponse(unsigned int cls,
                               NodeId_T responseId,
                               bool update = true);
    NodeId_T getEarlierNeuronId() const
    {
        return mEarlierId;
    };
    NodeId_T getMostActiveNeuronId() const
    {
        return mMostActiveId;
    };
    unsigned int getMostActiveNeuronRate() const
    {
        return mMostActiveRate;
    };
    unsigned int getFiringRate(NodeId_T nodeId) const;
    unsigned int getFiringRate(NodeId_T nodeId, EventType_T type) const
    {
        return mFiringRate.at(nodeId).at(type);
    };
    unsigned int getTotalFiringRate() const;
    unsigned int getTotalFiringRate(EventType_T type) const;
    unsigned int getNbNodes() const
    {
        return mNodes.size();
    };
    unsigned int getTotalActivity() const
    {
        return mTotalActivity;
    };
    double getSuccessRate(unsigned int avgWindow = 0) const;
    void logSuccessRate(const std::string& fileName,
                        unsigned int avgWindow = 0,
                        bool plot = false) const;
    /// Create a file to store firing rates and plot them if demanded by
    /// generating a gnuplot file.
    void logFiringRate(const std::string& fileName, bool plot = false) const;
    /// Create a file to store activities and plot them if demanded by
    /// generating a gnuplot file.
    void logActivity(const std::string& fileName,
                     bool plot = false,
                     Time_T start = 0,
                     Time_T stop = 0) const;
    /// Clear all (activity, firing rates and success)
    void clearAll();
    void clearActivity();
    void clearFiringRate();
    void clearSuccess();
    virtual ~Monitor() {};

    template <class T>
    static void logDataRate(const std::deque<T>& data,
                            const std::string& fileName,
                            unsigned int avgWindow = 0,
                            bool plot = false);

protected:
    /// The network that is monitored.
    Network& mNet;
    /// A vector of pointers to nodes to be recorded
    std::vector<Node*> mNodes;
    /// A map of spikes records arrays to corresponding neurons' IDs.
    std::map<NodeId_T, NodeEvents_T> mActivity;
    std::set<EventType_T> mRecordEventTypes;
    std::set<EventType_T> mEventTypes;
    std::map<NodeId_T, std::map<unsigned int, unsigned int> > mStats;
    /// Total number of spikes of each neuron.
    std::map<NodeId_T, std::map<EventType_T, unsigned int> > mFiringRate;
    std::deque<bool> mSuccess;
    /// The first neuron to spike (since last update).
    NodeId_T mEarlierId;
    /// The ID of the most active neuron (since last update).
    NodeId_T mMostActiveId;
    /// The rate of the most active neuron (since last update).
    unsigned int mMostActiveRate;
    /// The total number of spikes from all recorded neurons (since last
    /// update).
    unsigned int mTotalActivity;
    Time_T mLastEvent;
    Time_T mFirstEvent;
    bool mValidFirstEvent;
};
}

template <class T> void N2D2::Monitor::add(const std::vector<T*>& nodes)
{
    mNodes.insert(mNodes.end(), nodes.begin(), nodes.end());
    std::for_each(
        nodes.begin(),
        nodes.end(),
        std::bind(&T::setActivityRecording, std::placeholders::_1, true));
}

template <class T>
void N2D2::Monitor::logDataRate(const std::deque<T>& data,
                                const std::string& fileName,
                                unsigned int avgWindow,
                                bool plot)
{
    std::ofstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not create data rate log file: "
                                 + fileName);

    double accSuccess = 0.0;
    std::deque<T> accWindow;
    double accSum = 0.0;

    dataFile.precision(4);

    for (typename std::deque<T>::const_iterator it = data.begin(),
                                                itEnd = data.end();
         it != itEnd;
         ++it) {
        accSuccess += (*it);
        accWindow.push_back(*it);
        accSum += (*it);

        if (avgWindow > 0 && accWindow.size() > avgWindow) {
            accSum -= accWindow.front();
            accWindow.pop_front();
        }

        // Data file can become very big for deepnet, with >1,000,000 patterns,
        // that why we have to keep it minimal
        dataFile << (*it) << " " << accSum / (double)accWindow.size() << "\n";
    }

    dataFile.close();

    if (data.empty())
        std::cout << "Notice: no data rate recorded." << std::endl;
    else if (plot) {
        const double finalRate = (avgWindow > 0)
                                     ? accSum / (double)accWindow.size()
                                     : accSuccess / (double)data.size();

        std::ostringstream label;
        label << "\"Final: " << 100.0 * finalRate << "%";

        if (avgWindow > 0)
            label << " (last " << avgWindow << ")";

        label << "\" at graph 0.5, graph 0.1 front";

        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setYlabel("Success rate");
        gnuplot.setXlabel("# steps");
        gnuplot.setYrange(
            Utils::clamp(finalRate - (1.0 - finalRate), 0.0, 0.99),
            Utils::clamp(2.0 * finalRate, 0.01, 1.0));
        gnuplot.setY2range(0, 1);
        gnuplot.set("label", label.str());
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName,
                     "using ($0+1):2 with lines, \"\" using ($0+1):2 "
                     "with lines lc rgb \"light-gray\" axes x1y2");
    }
}

#endif // N2D2_MONITOR_H
