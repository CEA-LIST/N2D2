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
#ifdef CUDA

#include "CEnvironment_CUDA.hpp"

N2D2::CEnvironment_CUDA::CEnvironment_CUDA(Database& database,
                                         const std::vector<size_t>& size,
                                         unsigned int batchSize,
                                         bool compositeStimuli)
    : CEnvironment(database,
                    size,
                    batchSize,
                    compositeStimuli)

{
    // ctor

}


void N2D2::CEnvironment_CUDA::readBatch(Database::StimuliSet set,
                                      unsigned int startIndex)
{
    CEnvironment::readBatch(set, startIndex);
    mProvidedData[0].data.synchronizeHToD();

}


void N2D2::CEnvironment_CUDA::readRandomBatch(Database::StimuliSet set)
{
    StimuliProvider::readRandomBatch(set);
    mProvidedData[0].data.synchronizeHToD();
}


void N2D2::CEnvironment_CUDA::initialize()
{
    CEnvironment::initialize();
    
    if (0 == mNextEventTime.size()){ 
          
        mNextEventTime.resize(mProvidedData[0].data.dims());
        mNextEventType.resize(mProvidedData[0].data.dims());

        // Allocate global memory on device for curand states
        cudaMalloc((void **)&mCurandState, mProvidedData[0].data.dimB()*16*
                                sizeof(curandState));
        // Setup the initial curand states with the global seed value
        cudaSetupRng(mCurandState,
                    Random::_mt[0],
                    mProvidedData[0].data.dimB());
    }

    const cudaDeviceProp& deviceProp = CudaContext::getDeviceProp();
    mDeviceMaxThreads = (unsigned int) deviceProp.maxThreadsPerBlock;
}


//TODO: Check that this has save behavior as CEnvironment!
void N2D2::CEnvironment_CUDA::tick(Time_T timestamp, Time_T start, Time_T stop)
{
    
    if (mStopStimulus) {
        return;
    }
    if (mNoConversion) {
       
        cudaNoConversion(mProvidedData[0].data.getDevicePtr(),
                        mTickData.getDevicePtr(),
                        mTickActivity.getDevicePtr(),
                        mScaling,
                        mProvidedData[0].data.dimX(),
                        mProvidedData[0].data.dimY(),
                        mProvidedData[0].data.dimZ(),
                        mProvidedData[0].data.dimB(),
                        mDeviceMaxThreads);
        
        
        return;
    }

    SpikeGenerator::checkParameters();

    if (!mInitialized) {

       
        mProvidedData[0].data.synchronizeHToD();
        mNextEventTime.assign(mProvidedData[0].data.dims(), start);
        mNextEventType.assign(mProvidedData[0].data.dims(), 0);
        mTickData.assign(mProvidedData[0].data.dims(), 0);

        cudaGenerateInitialSpikes(mProvidedData[0].data.getDevicePtr(),
                            mNextEventTime.getDevicePtr(),
                            mNextEventType.getDevicePtr(),
                            mProvidedData[0].data.dimX(),
                            mProvidedData[0].data.dimY(),
                            mProvidedData[0].data.dimZ(),
                            start,
                            stop,
                            mDiscardedLateStimuli,
                            mStimulusType,
                            mPeriodMeanMin,
                            mPeriodMeanMax,
                            mPeriodRelStdDev,
                            mPeriodMin,
                            mMaxFrequency,
                            mProvidedData[0].data.dimB(),
                            mCurandState);
        // Setup the initial curand states with the global seed value
        cudaSetupRng(mCurandState,
                Random::_mt[0],
                mProvidedData[0].data.dimB());

        
        mInitialized = true;
    }

    cudaGenerateSpikes(mProvidedData[0].data.getDevicePtr(),
                        mTickData.getDevicePtr(),
                        //mTickOutputs.getDevicePtr(),
                        mNextEventTime.getDevicePtr(),
                        mNextEventType.getDevicePtr(),
                        mProvidedData[0].data.dimX(),
                        mProvidedData[0].data.dimY(),
                        mProvidedData[0].data.dimZ(),
                        timestamp,
                        start,
                        stop,
                        mDiscardedLateStimuli,
                        mStimulusType,
                        mPeriodMeanMin,
                        mPeriodMeanMax,
                        mPeriodRelStdDev,
                        mPeriodMin,
                        mMaxFrequency,
                        mProvidedData[0].data.dimB(),
                        mCurandState);
    
}

void N2D2::CEnvironment_CUDA::reset(Time_T /*timestamp*/)
{
    mInitialized = false;
    
    mTickActivity.assign(mTickActivity.dims(), 0);

    mStopStimulus = false;
}


N2D2::CEnvironment_CUDA::~CEnvironment_CUDA()
{
    cudaFree(mCurandState);
    // dtor
}

#endif

