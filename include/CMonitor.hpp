/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes Thiele (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
                 
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

#include "Network.hpp"
#include "FloatT.hpp"
#include "containers/Tensor.hpp"
#include "controler/Interface.hpp"
#include "utils/Gnuplot.hpp"


#ifdef CUDA
#include "containers/CudaTensor.hpp"
#include "controler/CudaInterface.hpp"
#endif



namespace N2D2 {
/**
 * The CMonitor class provides tools to monitor the activity of the network.
 * It keep track of the events emitted by a CSpike model.
 * Its aim is also to facilitate visual representation of the network's state
 * and its evolution.
*/
class CMonitor {
public:
    CMonitor();

    virtual void add(Tensor<float>& input);

    virtual void initialize(unsigned int nbTimesteps);
    virtual bool tick(Time_T timestamp);

    const Tensor<long long unsigned int>& getFiringRate() const
    {
        return mFiringRate;
    }

    const Tensor<long long unsigned int>& getTotalFiringRate() const
    {
        return mTotalFiringRate;
    }

    void clearAccumulators();

    const Tensor<long long int>& getOutputsActivity() const
    {
        return mOutputsActivity;
    }

    const Tensor<long long int>& getTotalOutputsActivity() const
    {
        return mOutputsActivity;
    }

    virtual long long unsigned int getIntegratedFiringRate() const
    {
        return std::accumulate(mFiringRate.begin(), mFiringRate.end(), 0);
    }

    virtual long long int getIntegratedOutputsActivity() const
    {
        return std::accumulate(mOutputsActivity.begin(), mOutputsActivity.end(), 0);
    }


    /// Create a file to store firing rates and plot them if demanded by
    /// generating a gnuplot file.
    void logFiringRate(const std::string& fileName, bool plot = false);
    void logTotalFiringRate(const std::string& fileName, bool plot = false);

    /// Create a file to store activities and plot them if demanded by
    /// generating a gnuplot file.
    void logActivity(const std::string& fileName,
                     unsigned int batch,
                     bool plot) const;

    /// Clear all (activity, firing rates and success)
    virtual void clearAll();
    virtual void clearActivity();
    virtual void reset(Time_T timestamp);

    virtual ~CMonitor() {};



protected:

#ifdef CUDA
    CudaTensor<float>* mInputs;

    CudaInterface<int> mActivity;

    /// Firing rates of each neuron over all examples
    CudaTensor<long long unsigned int> mFiringRate;
    /// Firing rates of each neuron for current example
    CudaTensor<long long unsigned int> mTotalFiringRate;

    /// Accumulated output of each neuron for current example
    CudaTensor<long long int> mOutputsActivity;
    /// Accumulated output of each neuron for all examples
    CudaTensor<long long int> mTotalOutputsActivity;

#else
    Tensor<float>* mInputs;

    Interface<int> mActivity;

    /// Firing rates of each neuron over all examples
    Tensor<long long unsigned int> mFiringRate;
    /// Firing rates of each neuron for current example
    Tensor<long long unsigned int> mTotalFiringRate;

    /// Accumulated output of each neuron for current example
    Tensor<long long int> mOutputsActivity;
    /// Accumulated output of each neuron for all examples
    Tensor<long long int> mTotalOutputsActivity;

#endif
  
    unsigned int mRelTimeIndex;
    unsigned int mNbTimesteps;
    bool mInitialized;

    std::map<Time_T, unsigned int> mTimeIndex;

};
}

#endif // N2D2_CMONITOR_H
