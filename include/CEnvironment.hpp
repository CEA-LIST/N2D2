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
                 unsigned int nbSubStimuli = 1,
                 bool compositeStimuli = false);
    void addChannel(const CompositeTransformation& transformation,
                    unsigned int subIdx=0);
    void setBatchSize(unsigned int batchSize);

    virtual void readBatch(Database::StimuliSet set,
                                      unsigned int startIndex);

    virtual void tick(Time_T timestamp, Time_T start, Time_T stop);

    virtual void readStimulus(Database::StimulusID id,
                      Database::StimuliSet set,
                      unsigned int batchPos = 0);

    virtual void loadAerStream(Time_T start,
                        Time_T stop);

    void readRelationalStimulus();

    void readRelationalStimulus(bool test,
                                bool sleep=false,
                                Float_T constantInput=0);

    void readDatabaseRelationalStimulus(Database::StimuliSet set);

    /*
    Tensor4d<Float_T> makeInputIdentity(unsigned int subStimulus,
                                        double scalingFactor);
    void produceConstantInput(Float_T constantInput);
    void produceRandomInput(Float_T mean, Float_T dev);*/
    double getRelationalTarget(unsigned int subStimulus)
    {
        return mRelationalTargets(subStimulus);
    };
    const Tensor<Float_T>& getRelationalTargets() const
    {
        return mRelationalTargets;
    };
    Interface<Float_T>& getRelationalData()
    {
        return mRelationalData;
    };
    const Tensor<Float_T>& getRelationalData(unsigned int subIdx) const
    {
        return mRelationalData[subIdx];
    };
    Tensor<Float_T>& getRelationalData(unsigned int subIdx)
    {
        return mRelationalData[subIdx];
    };
    const Tensor<Float_T>& getTickDataTrace(unsigned int subIdx) const
    {
        return mTickDataTraces[subIdx];
    };
    Tensor<Float_T>& getTickDataTrace(unsigned int subIdx)
    {
        return mTickDataTraces[subIdx];
    };
     const Tensor<Float_T>& getTickDataTraceLearning(unsigned int subIdx) const
    {
        return mTickDataTracesLearning[subIdx];
    };
    Tensor<Float_T>& getTickDataTraceLearning(unsigned int subIdx)
    {
        return mTickDataTracesLearning[subIdx];
    };
    bool doesSpikeConversion()
    {
        return !mNoConversion;
    };

    virtual void reset(Time_T timestamp);
    virtual void initialize();
    virtual void initializeSpikeGenerator(Time_T start, Time_T stop);

    virtual Tensor<int>& getTickData(unsigned int subIdx)
    {
        return mTickData[subIdx];
    };
    virtual const Tensor<int>& getTickData(unsigned int subIdx) const
    {
        return mTickData[subIdx];
    };
    virtual Interface<int>& getTickData()
    {
        return mTickData;
    };
    virtual Interface<Float_T>& getCurrentFiringRate()
    {
        return mCurrentFiringRate;
    };
    virtual Interface<Float_T>& getAccumulatedTickOutputs()
    {
        return mAccumulatedTickOutputs;
    };
    virtual Tensor<int>& getTickOutputs()
    {
        return mTickOutputs;
    };
    bool isAerMode()
    {
        return dynamic_cast<AER_Database*>(&mDatabase);
    };

    unsigned int getNbSubStimuli()
    {
        return mNbSubStimuli;
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
    CudaInterface<Float_T> mRelationalData;
#else
    Interface<Float_T> mRelationalData;
#endif

    Tensor<Float_T> mRelationalTargets;

#ifdef CUDA
    CudaInterface<int> mTickData;
    CudaTensor<int> mTickOutputs;
    CudaInterface<Float_T> mTickDataTraces;
    CudaInterface<Float_T> mTickDataTracesLearning;
    CudaInterface<Float_T> mCurrentFiringRate;
    CudaInterface<Float_T> mAccumulatedTickOutputs;
#else
    Interface<int> mTickData;
    Tensor<int> mTickOutputs;
    Interface<Float_T> mTickDataTraces;
    Interface<Float_T> mTickDataTracesLearning;
    Interface<Float_T> mCurrentFiringRate;
    Interface<Float_T> mAccumulatedTickOutputs;
#endif
    Interface<std::pair<Time_T, int> > mNextEvent;

    // With this iterator we avoid to iterate over all events in every tick
    std::vector<AerReadEvent>::iterator mEventIterator;
    Time_T mNextAerEventTime;

    Parameter<bool> mNoConversion;
    Parameter<Float_T> mScaling;
    Parameter<Time_T> mStopStimulusTime;
    Parameter<std::string> mStreamPath;
    unsigned int mNbSubStimuli;
    long long unsigned int mLastTimestamp;
    bool mStopStimulus;

};
}

#endif // N2D2_CENVIRONMENT_H
