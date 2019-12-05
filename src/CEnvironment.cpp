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

#include "CEnvironment.hpp"
#include "Database/MNIST_IDX_Database.hpp"

N2D2::CEnvironment::CEnvironment(Database& database,
                                 const std::vector<size_t>& size,
                                 unsigned int batchSize,
                                 unsigned int nbSubStimuli,
                                 bool compositeStimuli)
    : StimuliProvider(database, size, batchSize, compositeStimuli),
      SpikeGenerator(),
      mInitialized(false),
      mTickOutputs({size[0], size[1], size[2]*nbSubStimuli, batchSize}),
      mNextAerEventTime(0),
      mNoConversion(this, "NoConversion", false),
      mScaling(this, "Scaling", 1.0),
      mStopStimulusTime(this, "StopStimulusTime", 0),
      mStreamPath(this, "StreamPath", ""),
      mNbSubStimuli(nbSubStimuli)
{
   //ctor
    std::vector<size_t> dims({getSizeX(), getSizeY(), getNbChannels(), getBatchSize()});
    for (unsigned int k=0; k<nbSubStimuli; k++){
#ifdef CUDA
        mTickData.push_back(new CudaTensor<int>(dims));
        mTickDataTraces.push_back(new CudaTensor<Float_T>(dims));
        mTickDataTracesLearning.push_back(new CudaTensor<Float_T>(dims));
        mCurrentFiringRate.push_back(new CudaTensor<Float_T>(dims));
        mAccumulatedTickOutputs.push_back(new CudaTensor<Float_T>(dims));

#else
        mTickData.push_back(new Tensor<int>(dims));
        mTickDataTraces.push_back(new Tensor<Float_T>(dims));
        mTickDataTracesLearning.push_back(new Tensor<Float_T>(dims));
        mCurrentFiringRate.push_back(new Tensor<Float_T>(dims));
        mAccumulatedTickOutputs.push_back(new Tensor<Float_T>(dims));
#endif
        mNextEvent.push_back(new Tensor<std::pair<Time_T, int>>(dims));
    }
}


void N2D2::CEnvironment::initialize()
{
     // ctor
    std::vector<size_t> dims({getSizeX(), getSizeY(), getNbChannels(), getBatchSize()});
    mRelationalTargets.resize({mNbSubStimuli});

    if (mRelationalData.empty()) {
        if (mNbSubStimuli > 1) {
            for (unsigned int k=0; k<mNbSubStimuli; k++){
#ifdef CUDA
                mRelationalData.push_back(new CudaTensor<Float_T>(dims));
#else
                mRelationalData.push_back(new Tensor<Float_T>(dims));
#endif
            }
        }
        else {
            mRelationalData.push_back(&mData);
        }
    }

    mStopStimulus = false;
}


// TODO: Not tested yet!
void N2D2::CEnvironment::addChannel(const CompositeTransformation
                                    & transformation, unsigned int subIdx)
{
    if (!mChannelsTransformations.empty()) {
        mTickData[subIdx].resize(
                                {mTickData[subIdx].dimX(),
                                 mTickData[subIdx].dimY(),
                                 mTickData[subIdx].dimZ() + 1,
                                 mTickData[subIdx].dimB()});
        mNextEvent[subIdx].resize(
                                {mNextEvent[subIdx].dimX(),
                                 mNextEvent[subIdx].dimY(),
                                 mNextEvent[subIdx].dimZ() + 1,
                                 mNextEvent[subIdx].dimB()});
    }
    else {
        mTickData[subIdx].resize(
            {mTickData[subIdx].dimX(),
            mTickData[subIdx].dimY(),
            1,
            mTickData[subIdx].dimB()});
        mNextEvent[subIdx].resize(
            {mNextEvent[subIdx].dimX(),
            mNextEvent[subIdx].dimY(),
            1,
            mNextEvent[subIdx].dimB()});
    }

    StimuliProvider::addChannel(transformation);
}

void N2D2::CEnvironment::setBatchSize(unsigned int batchSize)
{
    mBatchSize = batchSize;

    if (mBatchSize > 0) {
        std::vector<size_t> dataSize(mData.dims());
        dataSize.back() = mBatchSize;

        mData.resize(dataSize);
        mFutureData.resize(dataSize);

        std::vector<size_t> labelSize(mLabelsData.dims());
        labelSize.back() = mBatchSize;

        mLabelsData.resize(labelSize);
        mFutureLabelsData.resize(labelSize);

        std::vector<size_t> dims({getSizeX(), getSizeY(), getNbChannels(), getBatchSize()});
        for (unsigned int k=0; k<mNbSubStimuli; k++){
            mTickData[k].resize(dims);
            mTickDataTraces[k].resize(dims);
            mTickDataTracesLearning[k].resize(dims);
            mCurrentFiringRate[k].resize(dims);
            mAccumulatedTickOutputs[k].resize(dims);
        }
    }

}


void N2D2::CEnvironment::readBatch(Database::StimuliSet set,
                                      unsigned int startIndex)
{
    const unsigned int batchSize
        = std::min(mBatchSize, mDatabase.getNbStimuli(set) - startIndex);

    // Fill mData batch elements which are not used with 0
    if (batchSize < mBatchSize) {
        std::fill(mRelationalData[0].begin(), mRelationalData[0].end(), 0);
    }

    std::vector<int>& batchRef = (mFuture) ? mFutureBatch : mBatch;

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        batchRef[batchPos]
            = mDatabase.getStimulusID(set, startIndex + batchPos);

#pragma omp parallel for schedule(dynamic) if (batchSize > 1)
    for (int batchPos = 0; batchPos < (int)batchSize; ++batchPos)
        readStimulus(batchRef[batchPos], set, batchPos);

    std::fill(batchRef.begin() + batchSize, batchRef.end(), -1);
}

void N2D2::CEnvironment::tick(Time_T timestamp, Time_T start, Time_T stop)
{
    if (mStopStimulusTime != 0 && timestamp > mStopStimulusTime * TimeNs + start) {
        if (!mStopStimulus) {
            mStopStimulus = true;
            clearTickOutput();
        }
        return;
    }
    if (mNoConversion) {
        for (unsigned int k=0; k<mRelationalData.size(); k++){
            //mTickDataTraceLearning[k].synchronizeDToH();
            //mRelationalData[k].synchronizeDToH();
            for (unsigned int idx = 0, size = mRelationalData[k].size(); idx < size; ++idx) {
                mTickDataTraces[k](idx) = mScaling*mRelationalData[k](idx);
                mTickDataTracesLearning[k](idx) += mTickDataTraces[k](idx);
            }
#ifdef CUDA
            mTickDataTracesLearning[k].synchronizeHToD();
            mTickDataTraces[k].synchronizeHToD();
            //std::cout << mRelationalData[k] << std::endl;
#endif
        }
        return;
    }

    AER_Database * aerDatabase = dynamic_cast<AER_Database*>(&mDatabase);
    if (aerDatabase) {

        SpikeGenerator::checkParameters();

        for (unsigned int k=0; k<mRelationalData.size(); k++){
            for (unsigned int idx = 0, size = mRelationalData[k].size(); idx < size; ++idx) {
                // If next event is valid set mTickData to spiking and search next event,
                // else set to non spiking
                if (mNextEvent[k](idx).second != 0 && mNextEvent[k](idx).first <= timestamp) {
                    std::pair<Time_T, int> event;
                    mTickData[k](idx) = mNextEvent[k](idx).second;


                    // Search for next event and check how many events are emitted in time window
                    for (unsigned int ev = 0; mNextEvent[k](idx).second != 0 &&
                                             mNextEvent[k](idx).first <= timestamp; ++ev) {
                        // If ev>0 a spike is lost
                        if (ev > 0) {
                            std::cout << Utils::cwarning << "cenv(" << idx
                                      << "): lost spike (scheduled @ "
                                      << mNextEvent[k](idx).first << ", previous @ "
                                      << event.first << ") @ " << timestamp
                                      << Utils::cdef << std::endl;
                        }

                        event = mNextEvent[k](idx);
                        SpikeGenerator::nextEvent(
                            mNextEvent[k](idx), mRelationalData[k](idx), start, stop);

                    }

                }
                else {
                    mTickData[k](idx) = 0;
                }
                //std::cout << idx <<  " data: " << mRelationalData[k](idx);
                //std::cout << " tickdata: " << (int)mTickData[k](idx) << std::endl;


                if (mTickData[k](idx) != 0) {
                    // TODO: Make coherent with other cells which take sign into account
                    mCurrentFiringRate[k](idx) += 1.0;
                }
                mAccumulatedTickOutputs[k](idx) += (int)mTickData[k](idx);
                mTickDataTraces[k](idx) = (int)mTickData[k](idx);
                mTickDataTracesLearning[k](idx) += (int)mTickData[k](idx);
            }
        }
    }
    else {

        if (mTickData.size() > 1) {
            throw std::runtime_error("TickData.size() > 1: SubStimuli not "
                                     "supported for event"
                                     " based databases yet");
        }


        mTickData[0].assign(
            mTickData[0].dims(), 0);

        while (mEventIterator != mAerData.end() &&
        (*mEventIterator).time + start <= timestamp) {
            unsigned int x = (*mEventIterator).x;
            unsigned int y = (*mEventIterator).y;
            unsigned int channel = (*mEventIterator).channel;

            if (x > mTickData[0].dimX() || y > mTickData[0].dimY()
            || channel > mTickData[0].dimZ()) {
                 throw std::runtime_error("Event coordinate out of range");
            }

            mTickData[0](x, y, channel, 0) = 1;
            mTickDataTraces[0](x, y, channel, 0) =
                 (int)mTickData[0](x, y, channel, 0);
            mTickDataTracesLearning[0](x, y, channel, 0) +=
                 (int)mTickData[0](x, y, channel, 0);
            ++mEventIterator;
        }
    }

    for (unsigned int k=0; k<mTickData.size(); k++){
        mTickDataTraces[k].synchronizeHToD();
        mTickDataTracesLearning[k].synchronizeHToD();
        mTickData[k].synchronizeHToD();
    }
     // TODO: This is currently only necessary for the CMonitor implementation
    for (unsigned int k=0; k<mTickData.size(); k++){
        for (unsigned int z=0; z<mTickData[k].dimZ(); ++z){
            for (unsigned int y=0; y<mTickData[k].dimY(); ++y){
                for (unsigned int x=0; x<mTickData[k].dimX(); ++x){
                    for (unsigned int batch=0; batch<mTickData.dimB(); ++batch){
                        mTickOutputs(x, y, z + k*mTickData[k].dimZ(), batch) =
                            mTickData[k](x, y, z, batch);
                    }
                }
            }
        }
    }
#ifdef CUDA
    mTickOutputs.synchronizeHToD();
#endif

}

// TODO: Adapt this to select subparts of mRelationalData
void N2D2::CEnvironment::readStimulus(Database::StimulusID id,
                                         Database::StimuliSet set,
                                         unsigned int batchPos)
{
    StimuliProvider::readStimulus(id, set, batchPos);

    if (mRelationalData.size() != 1){
        throw std::runtime_error("CEnvironment::readStimulus: "
                                 "mRelationalData.size() != 1");
    }
}


void N2D2::CEnvironment::loadAerStream(Time_T start,
                                        Time_T stop)
{
    mAerData.clear();

    std::ifstream data(mStreamPath);

    unsigned int x, y, polarity;
    unsigned int timestamp;

    if (data.good()) {
        // By default we use batch of size 1 and all spikes value 1
        while (data >> x >> y >> timestamp >> polarity){
            if ((start == 0 && stop == 0) || 
            (timestamp >= start && timestamp <= stop)){
                mAerData.push_back(AerReadEvent(x, y, polarity, 0, 1, 
                                                timestamp));
            }
        }
    }
    else {
        throw std::runtime_error("CEnvironment::loadAerStream: "
                                    "Could not open AER file: " + 
                                    std::string(mStreamPath));
    }

}


void N2D2::CEnvironment::readRelationalStimulus()
{
    // Note: this assumes at the moment that all sub stimuli have same size
    int stimulusSize = mRelationalData[0].size();
    double maxValue = 1.0*stimulusSize/100;
    unsigned int numberVariables = 3;
    std::vector<std::pair<std::vector<double>, double>> sample;
    for (unsigned int k=0; k<numberVariables; k++){

        std::pair<std::vector<double>, double> stimulusVariable;
        double variableValue = Random::randUniform(0.0, 1.0);

        if (k == numberVariables-1){
            double sum = 0;
            for (unsigned int p=0; p<numberVariables-1; p++){
                sum += sample[p].second;
            }
            variableValue = sum;
        }
        if (variableValue >= 1.0){
            variableValue = variableValue - 1.0;
        }
        stimulusVariable = std::make_pair(std::vector<double>(stimulusSize), variableValue);

        double slope = 0.02;//2*maxValue/stimulusSize;
        int centerVal = std::round(variableValue*stimulusSize);

        for (int x=0; x<stimulusSize; x++){
            double diffVal = (double)(x - centerVal);
            double yVal = maxValue - slope*std::fabs(diffVal);
            yVal =  yVal < 0 ? -yVal : yVal;
            stimulusVariable.first[x] = yVal;
        }
        std::cout << "Var: " << k << " : " << variableValue << std::endl;

        sample.push_back(stimulusVariable);
    }

    for (unsigned int k=0; k<sample.size(); k++){
        for (unsigned int i=0; i<sample[k].first.size(); ++i){
            mRelationalData[k](i) = sample[k].first[i];
        }
        mRelationalTargets(k) = sample[k].second;
        mRelationalData.back().synchronizeHToD();
    }

}

/// STDP based relational network
void N2D2::CEnvironment::readRelationalStimulus(bool test, bool sleep, Float_T constantInput)
{
    double maxValue = 1.0;
    unsigned int stimulusSize = mRelationalData[0].size();;
    unsigned int numberVariables = 3;
    std::vector<std::pair<std::vector<double>, double>> sample;
    for (unsigned int k=0; k<numberVariables; k++){

        std::pair<std::vector<double>, double> stimulusVariable;
        double variableValue = Random::randUniform(0.0, 1.0);

        if (k == numberVariables-1){
            double sum = 0;
            for (unsigned int p=0; p<numberVariables-1; p++){
                sum += sample[p].second;
            }
            variableValue = sum;
        }
        if (variableValue >= 1.0){
            variableValue = variableValue - 1.0;
        }
        stimulusVariable = std::make_pair(std::vector<double>(stimulusSize), variableValue);

        double slope = 2*maxValue/stimulusSize;
        double centerVal = variableValue*stimulusSize;

        for (unsigned int x=0; x<stimulusSize; x++){
            double diffVal = (double)x - centerVal;
            double yVal = maxValue - slope*std::fabs(diffVal);
            yVal =  yVal < 0 ? -yVal : yVal;
            stimulusVariable.first[x] = yVal;
        }

        sample.push_back(stimulusVariable);

    }


    unsigned int population = Random::randUniform(0,2);
    for (unsigned int k=0; k<sample.size(); k++){


        for (unsigned int i=0; i<sample[k].first.size(); ++i){
            if (sleep || (test && k==sample.size()-1)){
                mRelationalData[k](i) = Random::randNormal(constantInput,0.1);
                if (sleep && k != population){
                    mRelationalData[k](i) = 0.0;
                }
            }
            else {
                mRelationalData[k](i) = sample[k].first[i];
            }
        }
        mRelationalTargets(k) = sample[k].second;
        mRelationalData.back().synchronizeHToD();
    }

}



void N2D2::CEnvironment::readDatabaseRelationalStimulus(Database::StimuliSet set)
{
    Database * base = &mDatabase;
    MNIST_IDX_Database * relDatabase = dynamic_cast<MNIST_IDX_Database*>(base);

    if (relDatabase) {
        std::vector<unsigned int> sample
             = relDatabase->loadRelationSample(set);
        for (unsigned int k=0; k<sample.size(); k++){

            // TODO: Adapt to batch
            StimuliProvider::readStimulus(set, sample[k], 0);
            for (unsigned int i=0; i<mData.size(); ++i) {
                mRelationalData[k](i) = mData(i);
            }

            mRelationalTargets(k) = relDatabase->getStimulusLabel(set, sample[k]);
        }
        mRelationalData.back().synchronizeHToD();
    }
    else {
        throw std::runtime_error("CEnvironment::readRelationalStimulus: "
                                        "Dynamic cast failed!");
    }
}

/*
N2D2::Tensor4d<N2D2::Float_T> N2D2::CEnvironment::makeInputIdentity(unsigned int subStimulus,
                                                                    double scalingFactor)
{
    Tensor4d<Float_T> targetValues;
    targetValues.assign(mRelationalData[subStimulus].dimX(),
                      mRelationalData[subStimulus].dimY(),
                      mRelationalData[subStimulus].dimZ(),
                      mRelationalData[subStimulus].dimB(),
                      0);
    for (unsigned int idx = 0, size = mRelationalData[subStimulus].size(); idx < size; ++idx){
        targetValues(idx) = scalingFactor*mRelationalData[subStimulus](idx);
    }
    return targetValues;
}


void N2D2::CEnvironment::produceConstantInput(Float_T constantInput)
{
    for (unsigned int k = 0; k < mRelationalData.size(); ++k){
        for (unsigned int idx = 0, size = mRelationalData[k].size();
        idx < size; ++idx){
            mRelationalData[k](idx) = constantInput;
        }
        mRelationalData.back().synchronizeHToD();
    }

}


void N2D2::CEnvironment::produceRandomInput(Float_T mean, Float_T dev)
{
    for (unsigned int k = 0; k < mRelationalData.size(); ++k){
        for (unsigned int idx = 0, size = mRelationalData[k].size();
        idx < size; ++idx){
            Float_T randPixel = Random::randNormal(mean, dev);
            randPixel = randPixel < 0 ? 0 : randPixel;
            randPixel = randPixel > 1 ? 1 : randPixel;
            mRelationalData[k](idx) = randPixel;
        }
        mRelationalData.back().synchronizeHToD();
    }

}
*/


void N2D2::CEnvironment::reset(Time_T timestamp)
{
    for (unsigned int k=0; k<mRelationalData.size(); ++k){

        mTickData[k].assign(mTickData[k].dims(), 0);
        mTickDataTracesLearning[k].assign(mTickDataTracesLearning[k].dims(), 0);
        mCurrentFiringRate[k].assign(mCurrentFiringRate[k].dims(), 0);
        mAccumulatedTickOutputs[k].assign(mAccumulatedTickOutputs[k].dims(), 0);

        // This is usually overwritten immediately by initialize()
        mNextEvent[k].assign(mRelationalData[k].dims(),
                            std::make_pair(timestamp, 0));

    }

     mStopStimulus = false;
}


void N2D2::CEnvironment::initializeSpikeGenerator(Time_T start, Time_T stop)
{
    for (unsigned int k=0; k<mRelationalData.size(); ++k){

        for (unsigned int idx = 0, size = mRelationalData[k].size();
        idx < size; ++idx){
            SpikeGenerator::nextEvent(mNextEvent[k](idx),
                                      mRelationalData[k](idx),
                                      start,
                                      stop);
            //mTickData[k](idx) = mNextEvent[k](idx).second;
        }
    }

    AER_Database * aerDatabase = dynamic_cast<AER_Database*>(&mDatabase);
    if (aerDatabase) {
        mEventIterator = mAerData.begin();
    }
}

void N2D2::CEnvironment::clearTickOutput()
{
    for (unsigned int k=0; k<mRelationalData.size(); ++k){
        mTickData[k].assign(mTickData[k].dims(), 0);
        mTickDataTraces[k].assign(mTickDataTraces[k].dims(), 0);
    }
}

N2D2::CEnvironment::~CEnvironment()
{
    // dtor
}
