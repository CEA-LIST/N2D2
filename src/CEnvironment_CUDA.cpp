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
                                         unsigned int nbSubStimuli,
                                         bool compositeStimuli)
    : CEnvironment(database,
                    size,
                    batchSize,
                    nbSubStimuli,
                    compositeStimuli)

{
    // ctor

    for (unsigned int k=0; k<mRelationalData.size(); k++){
        mNextEventTime.push_back(new CudaTensor<Time_T>(
                                                mRelationalData[k].dims()));
        mNextEventType.push_back(new CudaTensor<char>(
                                                mRelationalData[k].dims()));
        curandState* state;
        // Allocate global memory on device for curand states
        cudaMalloc((void **)&state, mRelationalData[k].dimB()*16*
                                sizeof(curandState));
        // Setup the initial curand states with the global seed value
        cudaSetupRng(state,
                    Random::_mt[0],
                    mRelationalData[k].dimB());

        mCurandStates.push_back(state);
    }
    // TODO: Do we have to free this memory if the program terminates?

}


//TODO: Check that this has save behavior as CEnvironment!
void N2D2::CEnvironment_CUDA::tick(Time_T timestamp, Time_T start, Time_T stop)
{
    SpikeGenerator::checkParameters();

    if (!mInitialized) {

         mTickOutputs.assign(
                             {mRelationalData[0].dimX(),
                            mRelationalData[0].dimY(),
                            /*mRelationalData[0].dimZ()*mNbSubStimuli,*/
                            mRelationalData[0].dimZ(),
                            mRelationalData[0].dimB()}, 0);


        for (unsigned int k=0; k<mRelationalData.size(); k++){
            mRelationalData[k].synchronizeHToD();
            mNextEventTime[k].assign(mRelationalData[k].dims(), start);
            mNextEventType[k].assign(mRelationalData[k].dims(), 0);
            mTickData[k].assign(mRelationalData[k].dims(), 0);

            cudaGenerateInitialSpikes(mRelationalData[k].getDevicePtr(),
                                mNextEventTime[k].getDevicePtr(),
                                mNextEventType[k].getDevicePtr(),
                                mRelationalData[k].dimX(),
                                mRelationalData[k].dimY(),
                                mRelationalData[k].dimZ(),
                                start,
                                stop,
                                mDiscardedLateStimuli,
                                mStimulusType,
                                mPeriodMeanMin,
                                mPeriodMeanMax,
                                mPeriodRelStdDev,
                                mPeriodMin,
                                mRelationalData[k].dimB(),
                                mCurandStates[k]);
            // Setup the initial curand states with the global seed value
            cudaSetupRng(mCurandStates[k],
                    Random::_mt[0],
                    mRelationalData[k].dimB());

        }
        mInitialized = true;
    }

    for (unsigned int k=0; k<mRelationalData.size(); k++){
        cudaGenerateSpikes(mRelationalData[k].getDevicePtr(),
                            mTickData[k].getDevicePtr(),
                            mTickOutputs.getDevicePtr(),
                            mNextEventTime[k].getDevicePtr(),
                            mNextEventType[k].getDevicePtr(),
                            mRelationalData[k].dimX(),
                            mRelationalData[k].dimY(),
                            mRelationalData[k].dimZ(),
                            timestamp,
                            start,
                            stop,
                            mDiscardedLateStimuli,
                            mStimulusType,
                            mPeriodMeanMin,
                            mPeriodMeanMax,
                            mPeriodRelStdDev,
                            mPeriodMin,
                            mNbSubStimuli,
                            k,
                            mRelationalData[k].dimB(),
                            mCurandStates[k]);

    }

}

void N2D2::CEnvironment_CUDA::reset(Time_T /*timestamp*/)
{
    mInitialized = false;
}

N2D2::CEnvironment_CUDA::~CEnvironment_CUDA()
{
    // dtor
}

#endif

