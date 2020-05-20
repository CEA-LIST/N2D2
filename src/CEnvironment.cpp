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
                                 bool compositeStimuli)
    : StimuliProvider(database, size, batchSize, compositeStimuli),
      SpikeGenerator(),
      mInitialized(false),
      mNextAerEventTime(0),
      mNoConversion(this, "NoConversion", false),
      mScaling(this, "Scaling", 1.0),
      mStopStimulusTime(this, "StopStimulusTime", 0),
      mStreamPath(this, "StreamPath", "")
{
   //ctor
    std::vector<size_t> dims({getSizeX(), getSizeY(), getNbChannels(), getBatchSize()});

    mTickData.resize(dims);
    mTickActivity.resize(dims);
    mTickFiringRate.resize(dims);

    mNextEvent.resize(dims);
    
}


void N2D2::CEnvironment::initialize()
{
    //mInputData = &getData();
    //std::cout << mInputData << std::endl;
    mStopStimulus = false;
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
        
        mTickData.resize(dims);
        mTickActivity.resize(dims);
        mTickFiringRate.resize(dims);
        
    }

}


void N2D2::CEnvironment::readBatch(Database::StimuliSet set,
                                      unsigned int startIndex)
{
    const unsigned int batchSize
        = std::min(mBatchSize, mDatabase.getNbStimuli(set) - startIndex);

    // Fill mData batch elements which are not used with 0
    if (batchSize < mBatchSize) {
        std::fill(mData.begin(), mData.end(), 0);
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
            clearTickData();
        }
        return;
    }
    if (mNoConversion) {
        for (unsigned int idx = 0, size = mData.size(); idx < size; ++idx) {
            mTickData(idx) = mScaling*mData(idx);
            mTickActivity(idx) += mScaling*mTickData(idx);
        }
#ifdef CUDA
            mTickActivity.synchronizeHToD();
            mTickData.synchronizeHToD();
#endif
        return;
    }

    AER_Database * aerDatabase = dynamic_cast<AER_Database*>(&mDatabase);
    if (aerDatabase) {

        SpikeGenerator::checkParameters();
        for (unsigned int idx = 0, size = mData.size(); idx < size; ++idx) {
            // If next event is valid set mTickData to spiking and search next event,
            // else set to non spiking
            if (mNextEvent(idx).second != 0 && mNextEvent(idx).first <= timestamp) {
                std::pair<Time_T, int> event;
                mTickData(idx) = mNextEvent(idx).second;


                // Search for next event and check how many events are emitted in time window
                for (unsigned int ev = 0; mNextEvent(idx).second != 0 &&
                                            mNextEvent(idx).first <= timestamp; ++ev) {
                    // If ev>0 a spike is lost
                    if (ev > 0) {
                        std::cout << Utils::cwarning << "cenv(" << idx
                                    << "): lost spike (scheduled @ "
                                    << mNextEvent(idx).first << ", previous @ "
                                    << event.first << ") @ " << timestamp
                                    << Utils::cdef << std::endl;
                    }

                    event = mNextEvent(idx);
                    SpikeGenerator::nextEvent(
                        mNextEvent(idx), mData(idx), start, stop);

                }

            }
            else {
                mTickData(idx) = 0;
            }


            if (mTickData(idx) != 0) {
                mTickFiringRate(idx) += 1;
            }
            mTickActivity(idx) += mTickData(idx);
        }
        
    }
    else {

        
        mTickData.assign(
            mTickData.dims(), 0);

        while (mEventIterator != mAerData.end() &&
        (*mEventIterator).time + start <= timestamp) {
            unsigned int x = (*mEventIterator).x;
            unsigned int y = (*mEventIterator).y;
            unsigned int channel = (*mEventIterator).channel;

            if (x > mTickData.dimX() || y > mTickData.dimY()
            || channel > mTickData.dimZ()) {
                 throw std::runtime_error("Event coordinate out of range");
            }

            mTickData(x, y, channel, 0) = 1;
            mTickActivity(x, y, channel, 0) += mTickData(x, y, channel, 0);
            ++mEventIterator;
        }
    }

    
    mTickData.synchronizeHToD();
    mTickActivity.synchronizeHToD();
    
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


void N2D2::CEnvironment::reset(Time_T timestamp)
{


    mTickData.assign(mTickData.dims(), 0);
    mTickActivity.assign(mTickActivity.dims(), 0);
    mTickFiringRate.assign(mTickFiringRate.dims(), 0);

    // This is usually overwritten immediately by initialize()
    mNextEvent.assign(mData.dims(),
                        std::make_pair(timestamp, 0));

    mStopStimulus = false;
}


void N2D2::CEnvironment::initializeSpikeGenerator(Time_T start, Time_T stop)
{

    for (unsigned int idx = 0, size = mData.size();
    idx < size; ++idx){
        SpikeGenerator::nextEvent(mNextEvent(idx),
                                    mData(idx),
                                    start,
                                    stop);
        //mTickData[k](idx) = mNextEvent[k](idx).second;
    }

    AER_Database * aerDatabase = dynamic_cast<AER_Database*>(&mDatabase);
    if (aerDatabase) {
        mEventIterator = mAerData.begin();
    }
}

void N2D2::CEnvironment::clearTickData()
{
    mTickData.assign(mTickData.dims(), 0);
}

N2D2::CEnvironment::~CEnvironment()
{
    // dtor
}
