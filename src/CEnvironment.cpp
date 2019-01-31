/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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
      mReadAerData(this, "ReadAerData", false),
      mStreamPath(this, "StreamPath", ""),
      mNbSubStimuli(nbSubStimuli)
{
    // ctor
    mRelationalTargets.resize({mNbSubStimuli});
    for (unsigned int k=0; k<mNbSubStimuli; k++){
#ifdef CUDA
        mRelationalData.push_back(new CudaTensor<Float_T>({size[0],
                                                          size[1],
                                                          size[2],
                                                          batchSize}));
#else
        mRelationalData.push_back(new Tensor<Float_T>({size[0],
                                                      size[1],
                                                      size[2],
                                                      batchSize}));
#endif
    }
    for (unsigned int k=0; k<mRelationalData.size(); k++){
#ifdef CUDA
        mTickData.push_back(new CudaTensor<char>(mRelationalData[k].dims()));
        mCurrentFiringRate.push_back(new CudaTensor<Float_T>(mRelationalData[k].dims()));
        mAccumulatedTickOutputs.push_back(new CudaTensor<Float_T>(mRelationalData[k].dims()));

#else
        mTickData.push_back(new Tensor<char>(mRelationalData[k].dims()));
        mCurrentFiringRate.push_back(new Tensor<Float_T>(mRelationalData[k].dims()));
        mAccumulatedTickOutputs.push_back(new Tensor<Float_T>(mRelationalData[k].dims()));
#endif
        mNextEvent.push_back(new Tensor<std::pair<Time_T, char>>(
                                                    mRelationalData[k].dims()));
    }
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


void N2D2::CEnvironment::tick(Time_T timestamp, Time_T start, Time_T stop)
{
    if (!mReadAerData) {

        SpikeGenerator::checkParameters();

        for (unsigned int k=0; k<mRelationalData.size(); k++){
            for (unsigned int idx = 0, size = mRelationalData[k].size(); idx < size; ++idx) {
                // If next event is valid set mTickData to spiking and search next event,
                // else set to non spiking
                if (mNextEvent[k](idx).second != 0 && mNextEvent[k](idx).first <= timestamp) {
                    std::pair<Time_T, char> event;
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


                if (mTickData[k](idx) != 0) {
                    mCurrentFiringRate[k](idx) += 1.0;
                    mAccumulatedTickOutputs[k](idx) += (int)mTickData[k](idx);
                }
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
            ++mEventIterator;
        }

    }

    // TODO: This is currently only necessary for the CMonitor implementation
    for (unsigned int k=0; k<mTickData.size(); k++){
         mTickData[k].synchronizeHToD();
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

    if (mTickData.size() == 1){
        //mRelationalData.clear();
        for (unsigned int i=0; i<mData.size(); ++i) {
            mRelationalData[0](i) = mData(i);
        }
    }
}



void N2D2::CEnvironment::readRandomAerStimulus(Database::StimuliSet set,
                                        Time_T start,
                                        Time_T stop,
                                        unsigned int repetitions,
                                        unsigned int partialStimulus,
                                        std::vector<Database::StimulusID>& Ids)
{
    const Database::StimulusID id = getRandomID(set);
    readAerStimulus(set, id, start, stop, repetitions, partialStimulus);
    Ids.push_back(id);
}

void N2D2::CEnvironment::readAerStimulus(Database::StimuliSet set,
                                        Database::StimulusID id,
                                        Time_T start,
                                        Time_T stop,
                                        unsigned int repetitions,
                                        unsigned int partialStimulus,
                                        std::vector<Database::StimulusID>& Ids)
{
    readAerStimulus(set, id, start, stop, repetitions, partialStimulus);
    Ids.push_back(id);
}



void N2D2::CEnvironment::readAerStimulus(Database::StimuliSet set,
                                            Database::StimulusID id,
                                            Time_T start,
                                            Time_T stop,
                                            unsigned int repetitions,
                                            unsigned int partialStimulus)
{
    Database * base = &mDatabase;
    AER_Database * aerDatabase = dynamic_cast<AER_Database*>(base);


    if (aerDatabase) {
        //TODO: Implement getStimulusData in AER_Database and load into mData
        //std::cout << "Reading from AER database" << std::endl;
        std::vector<AerReadEvent> events =
            aerDatabase->loadAerStimulusData(set, id, start, stop, repetitions, partialStimulus);
        mAerData.clear();
        for (std::vector<AerReadEvent>::iterator it=events.begin();
        it!=events.end(); ++it) {
            mAerData.push_back(*it);
        }

        // TODO: Try to solve with swap
        //mAerData.swap(aerDatabase->loadAerStimulusData(id));
    }
}


void N2D2::CEnvironment::readAerStream(Time_T start,
                                    Time_T stop,
                                    unsigned int repetitions)
{
    readAerStream(mStreamPath, start, stop, repetitions);
}

void N2D2::CEnvironment::readAerStream(std::string dataPath,
                                            Time_T start,
                                            Time_T stop,
                                            unsigned int /*repetitions*/)
{
    mAerData.clear();

    std::ifstream data(dataPath);

    unsigned int x, y, polarity;
    unsigned int timestamp=0;

    if (data.good()) {
        while (data >> x >> y >> timestamp >> polarity){
            std::cout << start << " " << stop << " " << timestamp*TimeUs << std::endl;
            mAerData.push_back(AerReadEvent(x, y, 0, timestamp*TimeUs));
        }
    }
    else {
        throw std::runtime_error("CEnvironment::loadAerStream: "
                                    "Could not open AER file: " + dataPath);
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

        mTickData[k].assign(mRelationalData[k].dims(), 0);
        mCurrentFiringRate[k].assign(mRelationalData[k].dims(), 0);
        mAccumulatedTickOutputs[k].assign(mRelationalData[k].dims(), 0);

        // This is usually overwritten immediately by initialize()
        mNextEvent[k].assign(mRelationalData[k].dims(),
                            std::make_pair(timestamp, 0));

    }
}


void N2D2::CEnvironment::initialize(Time_T start, Time_T stop)
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

    if (mReadAerData) {
        mEventIterator = mAerData.begin();
    }
}

N2D2::CEnvironment::~CEnvironment()
{
    // dtor
}
