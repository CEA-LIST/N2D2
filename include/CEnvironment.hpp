/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
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

#ifndef N2D2_CENVIRONMENT_H
#define N2D2_CENVIRONMENT_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Network.hpp"
#include "SpikeGenerator.hpp"
#include "StimuliProvider.hpp"
#include "utils/Parameterizable.hpp"
#include "Database/AER_Database.hpp"

#ifdef CUDA
#include "controler/CudaInterface.hpp"
#else
#include "controler/Interface.hpp"
#endif

namespace N2D2 {
class CEnvironment : public StimuliProvider, public SpikeGenerator {
public:
    // Make other implementation of readStimulus accessible
    using StimuliProvider::readStimulus;
    CEnvironment(Database& database,
                 const std::vector<size_t>& size,
                 unsigned int batchSize = 1,
                 bool compositeStimuli = false);
 
    void setBatchSize(unsigned int batchSize);

    virtual void readBatch(Database::StimuliSet set,
                                      unsigned int startIndex);

    virtual void tick(Time_T timestamp, Time_T start, Time_T stop);

    virtual void readStimulus(Database::StimulusID id,
                      Database::StimuliSet set,
                      unsigned int batchPos = 0);

    virtual void loadAerStream(Time_T start,
                        Time_T stop);
    
    Tensor<Float_T>& getInputData()
    {
        return mInputData;
    };
   
    bool doesSpikeConversion()
    {
        return !mNoConversion;
    };

    virtual void reset(Time_T timestamp);
    virtual void initialize();
    virtual void initializeSpikeGenerator(Time_T start, Time_T stop);

    virtual Tensor<Float_T>& getTickData()
    {
        return mTickData;
    };
    
    virtual Tensor<Float_T>& getTickActivity()
    {
        return mTickActivity;
    };
   
    virtual Tensor<long long unsigned int>& getTickFiringRate()
    {
        return mTickFiringRate;
    };
   
    bool isAerMode()
    {
        return dynamic_cast<AER_Database*>(&mDatabase);
    };

    Time_T getStopStimulusTime()
    {
        return mStopStimulusTime;
    };

    void clearTickOutput();

    virtual ~CEnvironment();

protected:
    /// For each scale, tensor (x, y, channel, batch)
    bool mInitialized;

    std::vector<AerReadEvent> mAerData;

#ifdef CUDA
    // If CUDA activated use CudaTensor to enable CUDA spike generation
    CudaTensor<Float_T> mInputData;
#else
    Tensor<Float_T> mInputData;
#endif


#ifdef CUDA
    CudaTensor<Float_T> mTickData;
    //CudaTensor<int> mTickOutputs;
    CudaTensor<Float_T> mTickActivity;
    CudaTensor<long long unsigned int> mTickFiringRate;
#else
    Tensor<Float_T> mTickData;
    //Tensor<int> mTickOutputs;
    Tensor<Float_T> mTickActivity;
    Tensor<long long unsigned int> mTickFiringRate;
#endif
    Tensor<std::pair<Time_T, int> > mNextEvent;

    // With this iterator we avoid to iterate over all events in every tick
    std::vector<AerReadEvent>::iterator mEventIterator;
    Time_T mNextAerEventTime;

    Parameter<bool> mNoConversion;
    Parameter<Float_T> mScaling;
    Parameter<Time_T> mStopStimulusTime;
    Parameter<std::string> mStreamPath;
   
    long long unsigned int mLastTimestamp;
    bool mStopStimulus;

};
}

#endif // N2D2_CENVIRONMENT_H
