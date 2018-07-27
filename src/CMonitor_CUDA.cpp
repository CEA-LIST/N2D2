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


#ifdef CUDA

#include "CMonitor_CUDA.hpp"


N2D2::CMonitor_CUDA::CMonitor_CUDA(Network& net)
    : CMonitor(net),
    mCudaInput(true)

{
    // ctor
}

void N2D2::CMonitor_CUDA::add(StimuliProvider& sp)
{
    CEnvironment* cenvCSpike = dynamic_cast<CEnvironment*>(&sp);

    if (!cenvCSpike) {
          throw std::runtime_error(
            "CMonitor::add(): CMonitor models require CEnvironment");
    }

    mInputs.push_back<char>(&(cenvCSpike->getTickOutputs()));
    mInputs.back().setValid();


    // This is necessary to make CMonitor_CUDA compatible
    // with a non-Cuda CEnvironment
    CEnvironment_CUDA* cenv_cuda = dynamic_cast<CEnvironment_CUDA*>(&sp);

    if (!cenv_cuda) {
        mCudaInput = false;
    }

}


void N2D2::CMonitor_CUDA::add(Cell* cell)
{

    Cell_CSpike_CUDA* cellCSpike_CUDA = dynamic_cast<Cell_CSpike_CUDA*>(cell);

    if (cellCSpike_CUDA) {
        mInputs.push_back<char>(&(cellCSpike_CUDA->getOutputs()));

    }
    else {
         throw std::runtime_error("Error: CMonitor_CUDA could not add Cell."
            " Note: Cell has to be a CSpike and CUDA type.");
    }
    mInputs.back().setValid();
}


void N2D2::CMonitor_CUDA::initialize(unsigned int nbTimesteps,
                                    unsigned int nbClasses)
{
    if (!mInitialized) {
        mNbTimesteps = nbTimesteps;

        mNbClasses = nbClasses;

        mActivitySize = mInputs[0].dimX()* mInputs[0].dimY()
                              *mInputs[0].dimZ();

        mMostActiveId.resize({1, 1, 1, mInputs.dimB()}, 0);
        mMostActiveRate.resize({1, 1, 1, mInputs.dimB()}, 0);
        mFirstEventTime.resize({1, 1, 1, mInputs.dimB()}, 0);
        mLastEventTime.resize({1, 1, 1, mInputs.dimB()}, 0);


        mBatchActivity.resize({1, mInputs[0].dimX(), mInputs[0].dimY(),
                                mInputs[0].dimZ()}, 0);
        mTotalActivity.resize({1, 1, 1, mInputs.dimB()}, 0);

        mFiringRate.resize(mInputs[0].dims(), 0);
        mExampleActivity.resize(mInputs[0].dims(), 0);
        mLastExampleActivity.resize(mInputs[0].dims(), 0);
        mExampleIds.resize(mInputs[0].dims(), 0);
        const unsigned int channelSize = mInputs[0].dimX()*
            mInputs[0].dimY()* mInputs[0].dimZ();
        for(unsigned int batch = 0; batch < mInputs.dimB(); ++batch){
            for(unsigned int channel = 0; channel < channelSize; ++channel) {
                mExampleIds(channel, batch) = channel + batch*channelSize;
            }
        }
        mExampleIds.synchronizeHToD();

        mBatchFiringRate.resize({1, mInputs[0].dimX(), mInputs[0].dimY(),
                                mInputs[0].dimZ()}, 0);
        mTotalFiringRate.resize({1, 1, 1, mInputs.dimB()}, 0);

        for (unsigned int k=0; k<nbTimesteps; k++) {
            mActivity.push_back(new CudaTensor<char>(mInputs[0].dims()));
            mActivity.back().synchronizeHToD();

        }

        for (unsigned int k=0; k<nbClasses; k++) {
            mStats.push_back(new CudaTensor<unsigned int> (mInputs[0].dims()));
            mStats.back().synchronizeHToD();

        }
        mMaxClassResponse.resize(mInputs[0].dims(), 0);

        mInitialized = true;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        mDeviceMaxThreads = (unsigned int) deviceProp.maxThreadsPerBlock;
        mDeviceWarpSize = (unsigned int) deviceProp.warpSize;
    }
}


bool N2D2::CMonitor_CUDA::tick(Time_T timestamp)
{
    if (!mCudaInput){
        mInputs[0].synchronizeHToD();
    }
    cudaUpdateActivity(mInputs[0].getDevicePtr(),
                        mActivity[mRelTimeIndex].getDevicePtr(),
                        mFiringRate.getDevicePtr(),
                        mExampleActivity.getDevicePtr(),
                        mFirstEventTime.getDevicePtr(),
                        mLastEventTime.getDevicePtr(),
                        mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mInputs[0].dimZ(),
                        timestamp,
                        mInputs[0].dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);

    mTimeIndex.insert(std::make_pair(timestamp,mRelTimeIndex));
    mRelTimeIndex++;

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

    mTotalActivity.assign({1, 1, 1, mInputs.dimB()}, 0);
    mMostActiveRate.assign({1, 1, 1, mInputs.dimB()}, 0);


    cudaUpdateFiringRate(mFiringRate.getDevicePtr(),
                        mTotalFiringRate.getDevicePtr(),
                        mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mInputs[0].dimZ(),
                        mInputs[0].dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);

    cudaUpdateFiringRate(mExampleActivity.getDevicePtr(),
                        mTotalActivity.getDevicePtr(),
                        mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mInputs[0].dimZ(),
                        mInputs[0].dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);


    cudaUpdateBatchFiringRate(mFiringRate.getDevicePtr(),
                        mBatchFiringRate.getDevicePtr(),
                        mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mInputs[0].dimZ(),
                        mInputs[0].dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);

    cudaUpdateMostActive(mExampleIds.getDevicePtr(),
                        mExampleActivity.getDevicePtr(),
                        mMostActiveId.getDevicePtr(),
                        mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mInputs[0].dimZ(),
                        mInputs[0].dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);



    mFirstEventTime.synchronizeDToH();
    mLastEventTime.synchronizeDToH();
    for (unsigned int k=startIndex; k<=stopIndex; k++) {
        mActivity[k].synchronizeDToH();
    }
    mFiringRate.synchronizeDToH();
    mExampleActivity.synchronizeDToH();
    mTotalFiringRate.synchronizeDToH();
    mBatchFiringRate.synchronizeDToH();
    mTotalActivity.synchronizeDToH();
    mMostActiveId.synchronizeDToH();
    mTotalBatchFiringRate = 0;
    mTotalBatchActivity = 0;
    for(unsigned int batch=0; batch<mInputs[0].dimB(); ++batch) {
        mTotalBatchFiringRate += mTotalFiringRate(batch);
        mTotalBatchActivity += mTotalActivity(batch);
        // Remove the batch offset
        mMostActiveId(batch) -=
            batch*mInputs[0].dimX()*mInputs[0].dimY()*mInputs[0].dimZ();
        mMostActiveRate(batch) =
            mExampleActivity(mMostActiveId(batch),batch);

    }

    for (unsigned int i=0; i<mExampleActivity.size(); ++i){
        mLastExampleActivity(i) = mExampleActivity(i);
    }
    mExampleActivity.assign(mInputs[0].dims(), 0);

}

void N2D2::CMonitor_CUDA::clearAll(unsigned int nbTimesteps)
{

    mTotalBatchActivity = 0;
    mTotalBatchFiringRate = 0;
    mNbEvaluations = 0;
    mRelTimeIndex = 0;
    mSuccessCounter = 0;

    mActivitySize = mInputs[0].dimX()* mInputs[0].dimY()
                          *mInputs[0].dimZ();

    mMostActiveId.assign({1, 1, 1, mInputs.dimB()}, 0);
    mMostActiveRate.assign({1, 1, 1, mInputs.dimB()}, 0);
    mFirstEventTime.assign({1, 1, 1, mInputs.dimB()}, 0);
    mLastEventTime.assign({1, 1, 1, mInputs.dimB()}, 0);


    mBatchActivity.assign({1, mInputs[0].dimX(), mInputs[0].dimY(),
                            mInputs[0].dimZ()}, 0);
    mTotalActivity.assign({1, 1, 1, mInputs.dimB()}, 0);

    mFiringRate.assign(mInputs[0].dims(), 0);
    mBatchFiringRate.assign({1, mInputs[0].dimX(), mInputs[0].dimY(),
                            mInputs[0].dimZ()}, 0);
    mTotalFiringRate.assign({1, 1, 1, mInputs.dimB()}, 0);

    clearActivity(nbTimesteps);

    mSuccess.clear();
    clearFastSuccess();


    std::cout << "CMonitor_CUDA cleared" << std::endl;
}


void N2D2::CMonitor_CUDA::clearActivity(unsigned int nbTimesteps)
{
    //TODO: clear seems not properly defined
    //mActivity.clear();
    //for (unsigned int k=0; k<mNbTimesteps; k++) {
    //    mActivity.push_back(new CudaTensor<char>(mInputs[0].dimX(),
    //                mInputs[0].dimY(), mInputs[0].dimZ(), mInputs.dimB()));
    //}
    unsigned int oldNbTimesteps = mNbTimesteps;

    if (nbTimesteps != 0) {
        mNbTimesteps = nbTimesteps;
    }

    for (unsigned int k=0; k<mNbTimesteps; k++) {
        if (k >= oldNbTimesteps) {
             mActivity.push_back(new CudaTensor<char>(mInputs[0].dims()));
        }
        mActivity[k].assign(mInputs[0].dims(),0);
        mActivity.back().synchronizeHToD();
    }


    mTimeIndex.clear();
    mRelTimeIndex=0;

}





#endif
