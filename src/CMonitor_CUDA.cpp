/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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


#ifdef CUDA

#include "CMonitor_CUDA.hpp"


N2D2::CMonitor_CUDA::CMonitor_CUDA()
    : CMonitor()

{
    // ctor
}


void N2D2::CMonitor_CUDA::initialize(unsigned int nbTimesteps,
                                    unsigned int nbClasses)
{
    if (!mInitialized) {
        mNbTimesteps = nbTimesteps;

        mNbClasses = nbClasses;

        mActivitySize = (*mInputs).dimX()* (*mInputs).dimY()
                              *(*mInputs).dimZ();

        for (unsigned int k=0; k<nbTimesteps; k++) {
            mActivity.push_back(new CudaTensor<char>((*mInputs).dims()));
            mActivity.back().synchronizeHToD();

        }
        mBatchActivity.resize({1, (*mInputs).dimX(), (*mInputs).dimY(),
                                (*mInputs).dimZ()}, 0);

        mFiringRate.resize((*mInputs).dims(), 0);
        mTotalFiringRate.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mBatchFiringRate.resize({1, (*mInputs).dimX(), (*mInputs).dimY(),
                                (*mInputs).dimZ()}, 0);

        mExampleFiringRate.resize((*mInputs).dims(), 0);
        mTotalExampleFiringRate.resize({1, 1, 1, (*mInputs).dimB()}, 0);

        mOutputsActivity.resize((*mInputs).dims(), 0);
        mTotalOutputsActivity.resize({1, 1, 1, (*mInputs).dimB()}, 0);

        mMostActiveId.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mMostActiveRate.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mFirstEventTime.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mLastEventTime.resize({1, 1, 1, (*mInputs).dimB()}, 0);

        for (unsigned int k=0; k<nbClasses; k++) {
            mStats.push_back(new CudaTensor<unsigned int> ((*mInputs).dims()));
            mStats.back().synchronizeHToD();

        }

        mInitialized = true;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        mDeviceMaxThreads = (unsigned int) deviceProp.maxThreadsPerBlock;
        mDeviceWarpSize = (unsigned int) deviceProp.warpSize;
    }

}


bool N2D2::CMonitor_CUDA::tick(Time_T timestamp)
{
    mRelTimeIndex++;
    mTimeIndex.insert(std::make_pair(timestamp,mRelTimeIndex));

    cudaUpdateActivity((*mInputs).getDevicePtr(),
                        mActivity[mRelTimeIndex].getDevicePtr(),
                        mFiringRate.getDevicePtr(),
                        mExampleFiringRate.getDevicePtr(),
                        mOutputsActivity.getDevicePtr(),
                        mFirstEventTime.getDevicePtr(),
                        mLastEventTime.getDevicePtr(),
                        (*mInputs).dimX(),
                        (*mInputs).dimY(),
                        (*mInputs).dimZ(),
                        timestamp,
                        (*mInputs).dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);


    return false;
}

// This update function is based on a per example basis
// in contrast to the original Monitor implementation which
// is purely time based. This is due to the way how the activity
// is recorded and how it can be handled efficiently by CUDA
void N2D2::CMonitor_CUDA::update(Time_T start, Time_T stop)
{
    if (start > stop) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor_CUDA::getActivity: "
            "start > stop");
    }

    unsigned int startIndex;
    unsigned int stopIndex;
    if (start == 0 && stop == 0) {
        startIndex = 0;
        stopIndex = mTimeIndex.size()-1;
    }
    else {

        startIndex = mTimeIndex.at(start);
        stopIndex = mTimeIndex.at(stop);
    }

    // Reset example variables
    mMostActiveRate.assign(mMostActiveRate.dims(), 0);

    // Calculate metrics for each batch until next clear
    cudaUpdateFiringRate(mFiringRate.getDevicePtr(),
                        mTotalFiringRate.getDevicePtr(),
                        (*mInputs).dimX(),
                        (*mInputs).dimY(),
                        (*mInputs).dimZ(),
                        (*mInputs).dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);

    // Calculate example metrics for each batch
    cudaUpdateFiringRate(mExampleFiringRate.getDevicePtr(),
                        mTotalExampleFiringRate.getDevicePtr(),
                        (*mInputs).dimX(),
                        (*mInputs).dimY(),
                        (*mInputs).dimZ(),
                        (*mInputs).dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);

     // Calculate example metrics for each batch
    cudaUpdateFiringRate(mOutputsActivity.getDevicePtr(),
                        mTotalOutputsActivity.getDevicePtr(),
                        (*mInputs).dimX(),
                        (*mInputs).dimY(),
                        (*mInputs).dimZ(),
                        (*mInputs).dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);


    cudaUpdateBatchFiringRate(mFiringRate.getDevicePtr(),
                        mBatchFiringRate.getDevicePtr(),
                        (*mInputs).dimX(),
                        (*mInputs).dimY(),
                        (*mInputs).dimZ(),
                        (*mInputs).dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);


    cudaUpdateMostActive(mExampleFiringRate.getDevicePtr(),
                        mMostActiveId.getDevicePtr(),
                        (*mInputs).dimX(),
                        (*mInputs).dimY(),
                        (*mInputs).dimZ(),
                        (*mInputs).dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);



    mFirstEventTime.synchronizeDToH();
    mLastEventTime.synchronizeDToH();

    for (unsigned int k=startIndex; k<=stopIndex; k++) {
        mActivity[k].synchronizeDToH();
    }


    mFiringRate.synchronizeDToH();
    mTotalFiringRate.synchronizeDToH();
    mTotalExampleFiringRate.synchronizeDToH();
    mTotalOutputsActivity.synchronizeDToH();

    mExampleFiringRate.synchronizeDToH();
    //mBatchFiringRate.synchronizeDToH();
    mMostActiveId.synchronizeDToH();
    mTotalBatchFiringRate = 0;
    mTotalBatchExampleFiringRate = 0;
    mTotalBatchOutputsActivity = 0;



    for(unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
        mTotalBatchFiringRate += mTotalFiringRate(batch);
        mTotalBatchExampleFiringRate += mTotalExampleFiringRate(batch);
        mTotalBatchOutputsActivity += mTotalOutputsActivity(batch);
        // Remove the batch offset
        if (mMostActiveId(batch) >= (*mInputs).size()) {
            std::cout << "Most active ID: " << mMostActiveId(batch) <<
            "input: " << (*mInputs).size() << std::endl;
            exit(0);
        }

        mMostActiveRate(batch) =
            mExampleFiringRate(mMostActiveId(batch),batch);

    }

}







#endif
