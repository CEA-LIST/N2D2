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


void N2D2::CMonitor_CUDA::initialize(unsigned int nbTimesteps)
{
    if (!mInitialized) {
        mNbTimesteps = nbTimesteps;

        for (unsigned int k=0; k<nbTimesteps; k++) {
            mActivity.push_back(new CudaTensor<int>((*mInputs).dims()));
            mActivity.back().synchronizeHToD();

        }

        mFiringRate.resize((*mInputs).dims(), 0);
        mTotalFiringRate.resize((*mInputs).dims(), 0);

        mOutputsActivity.resize((*mInputs).dims(), 0);
        mTotalOutputsActivity.resize((*mInputs).dims(), 0);

        mInitialized = true;

        const cudaDeviceProp& deviceProp = CudaContext::getDeviceProp();
        mDeviceMaxThreads = (unsigned int) deviceProp.maxThreadsPerBlock;
        mDeviceWarpSize = (unsigned int) deviceProp.warpSize;
    }

}


bool N2D2::CMonitor_CUDA::tick(Time_T timestamp)
{
    cudaUpdateMetrics((*mInputs).getDevicePtr(),
                        mActivity[mRelTimeIndex].getDevicePtr(),
                        mFiringRate.getDevicePtr(),
                        mTotalFiringRate.getDevicePtr(),
                        mOutputsActivity.getDevicePtr(),
                        mTotalOutputsActivity.getDevicePtr(),
                        (*mInputs).dimX(),
                        (*mInputs).dimY(),
                        (*mInputs).dimZ(),
                        (*mInputs).dimB(),
                        mDeviceMaxThreads,
                        mDeviceWarpSize);

    mTimeIndex.insert(std::make_pair(timestamp,mRelTimeIndex));
    mRelTimeIndex++;

    if (mRelTimeIndex > mNbTimesteps){
        throw std::runtime_error("Error: more ticks than timesteps");
    }

    return false;
}



#endif
