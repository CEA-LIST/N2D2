/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes Thiele (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
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

#ifndef N2D2_CMONITOR_H
#define N2D2_CMONITOR_H

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
#include "Cell/Cell.hpp"
#include "CEnvironment.hpp"
#include "utils/Gnuplot.hpp"


#include "Cell/Cell_CSpike.hpp"
#ifdef CUDA
#include "containers/CudaTensor.hpp"
#include "controler/CudaInterface.hpp"
#endif



namespace N2D2 {
/**
 * The CMonitor class provides tools to monitor the activity of the network.
 * For example, it provides methods to automatically compute the recognition
 * rate of an unsupervised learning network.
 * Its aim is also to facilitate visual representation of the network's state
 * and its evolution.
*/
class CMonitor {
public:
    CMonitor(Network& net);
    virtual void add(StimuliProvider& sp);
    virtual void add(Cell* cell);

    virtual void initialize(unsigned int nbTimesteps, unsigned int nbClasses=0);
    virtual bool tick(Time_T timestamp);
    virtual void update(Time_T start, Time_T stop);

    /*bool classifyMajorityVote(unsigned int batch, unsigned int cls, bool update = true);
    bool classifyFirstSpike(unsigned int batch,
                            unsigned int cls,
                            unsigned int node,
                            bool update = true);*/
    bool classifyRateBased(int target,
                            Tensor<float> integrations,
                            unsigned int batch,
                            bool update=true);

    unsigned int classifyBatchRateBased(Tensor<int> targets,
                                        Tensor<float> integrations,
                                        bool update=true);

    bool classifyIntegrationBased(int target,
                                Tensor<float> integrations,
                                unsigned int batch,
                                bool update=true);

    unsigned int classifyBatchIntegrationBased(Tensor<int> targets,
                                        Tensor<float> integrations,
                                        bool update=true);

    bool checkLearningResponse(unsigned int batch, unsigned int cls,
                               bool update = true);
    unsigned int checkBatchLearningResponse(std::vector<unsigned int>& cls,
                               bool update = true);
    double inferRelation(unsigned int batch);
    void inferRelation(const std::string& fileName,
                       Tensor<Float_T>& relationalTargets,
                        unsigned int targetVariable,
                        unsigned int batch,
                        bool plot);

    void updateSuccess (bool success);

    NodeId_T getMostActiveNeuronId(unsigned int batch) const
    {
        return mMostActiveId(batch);
    };
    unsigned int getMostActiveNeuronRate(unsigned int batch) const
    {
        return mMostActiveRate(batch);
    };

    /// Activity -> per stimulus (batch) quantity
    unsigned int getActivitySize() const
    {
        return mActivitySize;
    };
    unsigned int getTotalBatchActivity() const
    {
        return mTotalBatchActivity;
    };
    unsigned int getTotalActivity(unsigned int batch) const
    {
        return mTotalActivity(batch);
    };
    unsigned int getActivity(unsigned int x,
                               unsigned int y,
                               unsigned int z,
                               unsigned int batch,
                               Time_T start=0,
                               Time_T stop=0);
    unsigned int getActivity(unsigned int index,
                             unsigned int batch,
                             Time_T start=0,
                             Time_T stop=0);
    unsigned int getBatchActivity(unsigned int x,
                               unsigned int y,
                               unsigned int z,
                               Time_T start=0,
                               Time_T stop=0);
    unsigned int getBatchActivity(unsigned int index,
                                  Time_T start=0,
                                  Time_T stop=0);

    virtual unsigned int calcTotalActivity(unsigned int batch,
                                    Time_T start=0,
                                    Time_T stop=0,
                                    bool update=false);
    unsigned int calcTotalBatchActivity(Time_T start=0,
                                        Time_T stop=0,
                                        bool update=false);

    /// Firing rate -> calculated above several stimuli (batches)
    unsigned int getTotalBatchFiringRate() const
    {
        return mTotalBatchFiringRate;
    }
    unsigned int getTotalFiringRate(unsigned int batch) const
    {
        return mTotalFiringRate(batch);
    }
    unsigned int getBatchFiringRate(unsigned int x,
                                   unsigned int y,
                                   unsigned int z) const
    {
        return mBatchFiringRate(0,x,y,z);
    }
#ifdef CUDA
    CudaTensor<unsigned int>& getFiringRate()
    {
        return mFiringRate;
    }
#else
    Tensor<unsigned int>& getFiringRate()
    {
        return mFiringRate;
    }
#endif

#ifdef CUDA
    CudaTensor<unsigned int>& getExampleActivity()
    {
        return mLastExampleActivity;
    }
#else
    Tensor<unsigned int>& getExampleActivity()
    {
        return mLastExampleActivity;
    }
#endif


    double getSuccessRate(unsigned int avgWindow = 0) const;
    double getFastSuccessRate() const;

    void logSuccessRate(const std::string& fileName,
                        unsigned int avgWindow = 0,
                        bool plot = false) const;
    /// Create a file to store firing rates and plot them if demanded by
    /// generating a gnuplot file.
    void logFiringRate(const std::string& fileName, bool plot = false,
                       Time_T start = 0, Time_T stop = 0) ;
    /// Create a file to store activities and plot them if demanded by
    /// generating a gnuplot file.
    void logActivity(const std::string& fileName,
                     unsigned int batch,
                     bool plot = false,
                     Time_T start = 0,
                     Time_T stop = 0) const;
    /// Clear all (activity, firing rates and success)
    virtual void clearAll(unsigned int nbTimesteps=0);
    virtual void clearActivity(unsigned int nbTimesteps=0);
    void clearFiringRate();
    void clearMostActive();
    void clearSuccess();
    void clearFastSuccess();
    virtual ~CMonitor() {};

    template <class T>
    static void logDataRate(const std::deque<T>& data,
                            const std::string& fileName,
                            unsigned int avgWindow = 0,
                            bool plot = false);
    template <class T>
    static void logErrorRate(const std::deque<T>& data,
                            const std::string& fileName,
                            unsigned int avgWindow = 0,
                            bool plot = false);
    template <class T>
    static void logPoints(const std::deque<T>& data,
                        const std::string& fileName,
                        const unsigned int logInterval,
                        bool plot = false);


protected:
    /// The network that is monitored.
    Network& mNet;

#ifdef CUDA
    CudaInterface<char> mInputs;

    CudaInterface<unsigned int> mStats;
    CudaTensor<unsigned int> mMaxClassResponse;

    /// Firing rates of each neuron in update phase
    CudaTensor<unsigned int> mFiringRate;
    CudaTensor<unsigned int> mBatchFiringRate;
    CudaTensor<unsigned int> mTotalFiringRate;


    /// The ID of the most active neuron (since last update).
    CudaTensor<NodeId_T> mMostActiveId;
    /// The rate of the most active neuron (since last update).
    CudaTensor<unsigned int> mMostActiveRate;
    /// The total number of spikes from all recorded neurons (since last
    /// update).

    CudaInterface<char> mActivity;
    CudaTensor<char> mBatchActivity;
    CudaTensor<unsigned int> mTotalActivity;

    CudaTensor<unsigned int> mLastExampleActivity;

    // TODO: This is not updated properly
    // In clock based simulation this will often be ambiguous
    CudaTensor<Time_T> mFirstEventTime;
    CudaTensor<Time_T> mLastEventTime;
#else
    Interface<char> mInputs;

    Interface<unsigned int> mStats;
    Tensor<unsigned int> mMaxClassResponse;

    /// Firing rates of each neuron in update phase
    Tensor<unsigned int> mFiringRate;
    Tensor<unsigned int> mBatchFiringRate;
    Tensor<unsigned int> mTotalFiringRate;


    /// The ID of the most active neuron (since last update).
    Tensor<NodeId_T> mMostActiveId;
    /// The rate of the most active neuron (since last update).
    Tensor<unsigned int> mMostActiveRate;
    /// The total number of spikes from all recorded neurons (since last
    /// update).

    Interface<char> mActivity;
    Tensor<char> mBatchActivity;
    Tensor<unsigned int> mTotalActivity;

    Tensor<unsigned int> mLastExampleActivity;

    Tensor<Time_T> mFirstEventTime;
    Tensor<Time_T> mLastEventTime;
#endif
    unsigned int mTotalBatchActivity;
    unsigned int mTotalBatchFiringRate;
    unsigned int mNbEvaluations;
    unsigned int mRelTimeIndex;
    unsigned int mSuccessCounter;
    unsigned int mNbTimesteps;
    unsigned int mNbClasses;
    unsigned int mActivitySize;
    bool mInitialized;


    std::map<Time_T, unsigned int> mTimeIndex;
    std::deque<bool> mSuccess;
    std::deque<Tensor<Float_T>> relationalInferences;

};
}


template <class T>
void N2D2::CMonitor::logDataRate(const std::deque<T>& data,
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


template <class T>
void N2D2::CMonitor::logErrorRate(const std::deque<T>& data,
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
        const double finalRate = std::sqrt((avgWindow > 0)
                                     ? accSum / (double)accWindow.size()
                                     : accSuccess / (double)data.size());

        std::ostringstream label;
        label << "\"RMSE: " << finalRate;

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

template <class T>
void N2D2::CMonitor::logPoints(const std::deque<T>& data,
            const std::string& fileName,
            const unsigned int logInterval,
            bool plot)
{
    std::ofstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not create data rate log file: "
                                 + fileName);

    double maxRate = 0.0;
    //double averageRate = 0.0;

    dataFile.precision(4);

    unsigned int i = 1;
    for (typename std::deque<T>::const_iterator it = data.begin(),
                                                itEnd = data.end();
         it != itEnd;
         ++it) {

        if (*it > maxRate) {
            maxRate = (double)*it;
        }
        //averageRate += (double)*it;
        // Data file can become very big for deepnet, with >1,000,000 patterns,
        // that why we have to keep it minimal
        dataFile << i*logInterval << " " << *it << "\n";
        i++;
    }

    dataFile.close();

    if (data.empty())
        std::cout << "Notice: no data rate recorded." << std::endl;
    else if (plot) {

        std::ostringstream label;
        label << "\"Maximal: " << maxRate;
        //label << " Average: " << maxRate/data.size();

        label << "\" at graph 0.5, graph 0.1 front";

        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setYlabel("");
        gnuplot.setXlabel("# steps");
        //gnuplot.setYrange(
        //    Utils::clamp(maxRate - (1.0 - maxRate), 0.0, 0.99),
        //    Utils::clamp(2.0 * maxRate, 0.01, 1.0));
        //gnuplot.setY2range(0, 1);
        gnuplot.set("label", label.str());
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName,
                     "using 1:2 with lines");
    }
}




#endif // N2D2_CMONITOR_H
