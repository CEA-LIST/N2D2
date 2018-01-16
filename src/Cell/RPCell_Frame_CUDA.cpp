/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
    
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

#include "Cell/RPCell_Frame_CUDA.hpp"
#include <thrust/device_ptr.h>
N2D2::Registrar<N2D2::RPCell>
N2D2::RPCell_Frame_CUDA::mRegistrar("Frame_CUDA", N2D2::RPCell_Frame_CUDA::create);

N2D2::RPCell_Frame_CUDA::RPCell_Frame_CUDA(const std::string& name,
                                 unsigned int nbAnchors,
                                 unsigned int nbProposals,
                                 unsigned int scoreIndex,
                                 unsigned int IoUIndex)
    : Cell(name, 4),
      RPCell(name, nbAnchors, nbProposals, scoreIndex, IoUIndex),
      Cell_Frame_CUDA(name, 4)
{
    // ctor
}

void N2D2::RPCell_Frame_CUDA::initialize()
{
    mAnchors.resize(mNbProposals * mInputs.dimB());
    unsigned int outputMaxSize = mNbAnchors*mInputs[0].dimY()*mInputs[0].dimX();
    unsigned int sortedSize = outputMaxSize;

    if(mPre_NMS_TopN > 0 && mPre_NMS_TopN < outputMaxSize)
        sortedSize = mPre_NMS_TopN;

    const int col_blocks = DIVUP(sortedSize, sizeof(unsigned long long) * 8);
        
    mValues.resize(1, 1, outputMaxSize, mInputs.dimB());    
    mIndexI.resize(1, 1, outputMaxSize, mInputs.dimB());
    mIndexJ.resize(1, 1, outputMaxSize, mInputs.dimB());
    mIndexK.resize(1, 1, outputMaxSize, mInputs.dimB());
    mIndexB.resize(1, 1, outputMaxSize, mInputs.dimB());
    mSortedIndexI.resize(1, 1, sortedSize, mInputs.dimB());
    mSortedIndexJ.resize(1, 1, sortedSize, mInputs.dimB());
    mSortedIndexK.resize(1, 1, sortedSize, mInputs.dimB());
    mSortedIndexB.resize(1, 1, sortedSize, mInputs.dimB());
    mMask.resize(1, 1, sortedSize*col_blocks, mInputs.dimB());
    mMap.resize(1, 1, outputMaxSize, mInputs.dimB());
    mSortedIndex.resize(1, 1, sortedSize, mInputs.dimB());
    mGPUAnchors.resize(1, 1, mNbProposals*4, mInputs.dimB());

    mMap.synchronizeHToD();    
    mValues.synchronizeHToD();    
    mIndexI.synchronizeHToD();
    mIndexJ.synchronizeHToD();
    mIndexK.synchronizeHToD();
    mIndexB.synchronizeHToD();
    mSortedIndexI.synchronizeHToD();
    mSortedIndexJ.synchronizeHToD();
    mSortedIndexK.synchronizeHToD();
    mSortedIndexB.synchronizeHToD();
    mMask.synchronizeHToD();
}

void N2D2::RPCell_Frame_CUDA::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();
    const unsigned int outputMaxSize = mNbAnchors*mInputs[0].dimY()*mInputs[0].dimX();
    const unsigned int nbBlocks = std::ceil(outputMaxSize/(float) 32.0);
    unsigned int sortedSize = outputMaxSize;
    
    if(mPre_NMS_TopN > 0 && mPre_NMS_TopN < outputMaxSize)
        sortedSize = mPre_NMS_TopN;

    /**Reorder i,j,k,b index and create the map vector to allow a fast gpu sorting using thrust**/
    cudaSSplitIndexes(  mInputs[0].dimX(),
                        mInputs[0].dimY(), 
                        mNbAnchors, 
                        mInputs[0].dimB(), 
                        nbBlocks,
                        mInputs[0].getDevicePtr(), 
                        mValues.getDevicePtr(),
                        mIndexI.getDevicePtr(), 
                        mIndexJ.getDevicePtr(), 
                        mIndexK.getDevicePtr(),
                        mIndexB.getDevicePtr(), 
                        mMap.getDevicePtr(), 
                        mMinWidth, 
                        mMinHeight, 
                        mScoreIndex);

    for(unsigned int n = 0; n < mInputs[0].dimB(); ++n)
    {
        unsigned int inputOffset = mMap.dimX()*mMap.dimY()*mMap.dimZ()*n;
        unsigned int outputOffset = mSortedIndexI.dimX()*mSortedIndexI.dimY()*mSortedIndexI.dimZ()*n;

        thrust_sort_keys(   mValues.getDevicePtr(), 
                            mMap.getDevicePtr(), 
                            outputMaxSize,
                            outputMaxSize*n);
        
        thrust_gather(  mMap.getDevicePtr(), 
                        mIndexI.getDevicePtr(), 
                        mSortedIndexI.getDevicePtr(), 
                        sortedSize,
                        inputOffset,
                        outputOffset);

        thrust_gather(  mMap.getDevicePtr(), 
                        mIndexJ.getDevicePtr(), 
                        mSortedIndexJ.getDevicePtr(), 
                        sortedSize,
                        inputOffset,
                        outputOffset);

        thrust_gather(  mMap.getDevicePtr(), 
                        mIndexK.getDevicePtr(), 
                        mSortedIndexK.getDevicePtr(), 
                        sortedSize,
                        inputOffset,
                        outputOffset);

        thrust_gather(  mMap.getDevicePtr(), 
                        mIndexB.getDevicePtr(), 
                        mSortedIndexB.getDevicePtr(), 
                        sortedSize,
                        inputOffset,
                        outputOffset);

    }

    if(inference)
    {
        const unsigned int nbThreadsPerBlocks = sizeof(unsigned long long) * 8;
        const int col_blocks = DIVUP(sortedSize, nbThreadsPerBlocks);
        dim3 blocks(DIVUP(sortedSize, nbThreadsPerBlocks), DIVUP(sortedSize, nbThreadsPerBlocks));

        dim3 threads(nbThreadsPerBlocks);

        for(unsigned int n = 0; n < mInputs[0].dimB(); ++n)
        {
            unsigned int inputOffset = n*mInputs[0].dimX()*mInputs[0].dimY()*mInputs[0].dimZ();
            unsigned int indexOffset = n*mSortedIndexI.dimX()*mSortedIndexI.dimY()*mSortedIndexI.dimZ();
            unsigned int outputOffset = n*mMask.dimX()*mMask.dimY()*mMask.dimZ();

            cudaSnms( mInputs[0].dimX(),
                    mInputs[0].dimY(), 
                    mNbAnchors, 
                    1, 
                    mInputs[0].getDevicePtr(), 
                    inputOffset,
                    mSortedIndexI.getDevicePtr(), 
                    mSortedIndexJ.getDevicePtr(), 
                    mSortedIndexK.getDevicePtr(), 
                    mSortedIndexB.getDevicePtr(), 
                    indexOffset,
                    mMask.getDevicePtr(),
                    outputOffset,
                    mNMS_IoU_Threshold,
                    sortedSize,
                    threads,
                    blocks);
        }

        mMask.synchronizeDToH();
               
        std::vector<std::vector<unsigned long long> > remv(mInputs[0].dimB(),
                                                           std::vector<unsigned long long>(col_blocks, 0));

        for(unsigned int n = 0; n < mInputs[0].dimB(); ++n)
        {
            int num_to_keep = 0;
            const unsigned int sortOffset = mSortedIndex.dimX()*mSortedIndex.dimY()*mSortedIndex.dimZ()*n;
            const unsigned int maskOffset = mMask.dimX()*mMask.dimY()*mMask.dimZ()*n;

            for (int i = 0; i < (int) sortedSize; i++) 
            {
                int nblock = i / nbThreadsPerBlocks;
                int inblock = i % nbThreadsPerBlocks;
        
                if (!(remv[n][nblock] & (1ULL << inblock))) 
                {
                    mSortedIndex(num_to_keep + sortOffset) = i;
                    //std::cout << "mSortedIndex(" << num_to_keep + sortOffset << "): " 
                    //    << i << " for n " << n << std::endl;
                    num_to_keep++;

                    unsigned long long *p = &mMask(0) + i * col_blocks + maskOffset;

                    for (int j = nblock; j < col_blocks; j++) 
                    {
                        remv[n][j] |= p[j];
                    }
                }
            }
        }

        mSortedIndex.synchronizeHToD();

        cudaSGatherRP( mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mNbAnchors,
                        mInputs[0].dimB(),
                        mInputs[0].getDevicePtr(), 
                        mSortedIndexI.getDevicePtr(), 
                        mSortedIndexJ.getDevicePtr(), 
                        mSortedIndexK.getDevicePtr(), 
                        mSortedIndexB.getDevicePtr(), 
                        mSortedIndex.getDevicePtr(),
                        mOutputs.getDevicePtr(),
                        mGPUAnchors.getDevicePtr(),
                        sortedSize,
                        mNbProposals,
                        (unsigned int) std::ceil(mNbProposals/(float)32));
        
        mGPUAnchors.synchronizeDToH();

        for(unsigned int anchorIdx = 0; anchorIdx < mNbProposals*mInputs[0].dimB(); ++anchorIdx)
        {
            mAnchors[anchorIdx].i = mGPUAnchors(0 + anchorIdx*4);
            mAnchors[anchorIdx].j = mGPUAnchors(1 + anchorIdx*4);
            mAnchors[anchorIdx].k = mGPUAnchors(2 + anchorIdx*4);
            mAnchors[anchorIdx].b = mGPUAnchors(3 + anchorIdx*4);                
        }
    }

    Cell_Frame_CUDA::propagate();
    
    mDiffInputs.clearValid();
}

void N2D2::RPCell_Frame_CUDA::backPropagate()
{
    // No backpropagation for this layer
}

void N2D2::RPCell_Frame_CUDA::update()
{
    // Nothing to update
}

void N2D2::RPCell_Frame_CUDA::setOutputsSize()
{
    RPCell::setOutputsSize();

    if (mOutputs.empty()) {
        mOutputs.resize(mOutputsWidth,
                        mOutputsHeight,
                        mNbOutputs,
                        mNbProposals * mInputs.dimB());
        mDiffInputs.resize(mOutputsWidth,
                           mOutputsHeight,
                           mNbOutputs,
                           mNbProposals * mInputs.dimB());
    }
}

#endif