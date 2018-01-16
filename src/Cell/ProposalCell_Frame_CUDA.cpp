/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
                    
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

#include "Cell/ProposalCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::ProposalCell>
N2D2::ProposalCell_Frame_CUDA::mRegistrar("Frame_CUDA", N2D2::ProposalCell_Frame_CUDA::create);

N2D2::ProposalCell_Frame_CUDA::ProposalCell_Frame_CUDA(const std::string& name,
                                            StimuliProvider& sp,
                                            unsigned int nbProposals,
                                            unsigned int scoreIndex,
                                            unsigned int IoUIndex,
                                            bool isNms,
                                            std::vector<double> meansFactor,
                                            std::vector<double> stdFactor)
    : Cell(name, 4),
      ProposalCell(name, sp, nbProposals, scoreIndex, IoUIndex, isNms, meansFactor, stdFactor),
      Cell_Frame_CUDA(name, 4)
{
    // ctor
}

void N2D2::ProposalCell_Frame_CUDA::initialize()
{
    const unsigned int inputBatch = mOutputs.dimB()/mNbProposals;

    mMeansCUDA.resize(4);
    mStdCUDA.resize(4);
    for(unsigned int i = 0 ; i < 4; ++i)
    {
        mMeansCUDA(i) = mMeanFactor[i];
        mStdCUDA(i) = mStdFactor[i];
    }

    mNormalizeROIs.resize(1, 1, 4, mNbProposals*inputBatch);

    mNormalizeROIs.synchronizeHToD();    
    mMeansCUDA.synchronizeHToD();    
    mStdCUDA.synchronizeHToD();    

}

void N2D2::ProposalCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();
    const Float_T normX = 1.0 / (mStimuliProvider.getSizeX() - 1) ;
    const Float_T normY = 1.0 / (mStimuliProvider.getSizeY() - 1) ;
    const unsigned int inputBatch = mOutputs.dimB()/mNbProposals;
    const unsigned int blockSize = std::ceil( (float)(mNbProposals) / (float)  32);
    const dim3 nbThread = {32, 1 , 1};
    const dim3 nbBlocks = {blockSize, 1 , inputBatch};

    cudaSNormalizeROIs( mInputs[0].dimX(),
                        mInputs[0].dimY(), 
                        mNbProposals, 
                        inputBatch, 
                        mScoreIndex,
                        normX,
                        normY,
                        mMeansCUDA.getDevicePtr(),
                        mStdCUDA.getDevicePtr(),
                        mInputs[0].getDevicePtr(), 
                        mInputs[1].getDevicePtr(),
                        mInputs[2].getDevicePtr(),
                        mNormalizeROIs.getDevicePtr(),
                        mScoreThreshold,
                        nbThread,
                        nbBlocks);

    if(mApplyNMS)
    {
        mNormalizeROIs.synchronizeDToH();
        // Non-Maximum Suppression (NMS)
        for(unsigned int n = 0; n < inputBatch; ++n)
        {
            for (unsigned int i = 0; i < mNbProposals - 1;
                ++i)
            {
                const Float_T x0 = mNormalizeROIs(0, i + n*mNbProposals);
                const Float_T y0 = mNormalizeROIs(1, i + n*mNbProposals);
                const Float_T w0 = mNormalizeROIs(2, i + n*mNbProposals);
                const Float_T h0 = mNormalizeROIs(3, i + n*mNbProposals);

                for (unsigned int j = i + 1; j < mNbProposals; ) {

                    const Float_T x = mNormalizeROIs(0, j + n*mNbProposals);
                    const Float_T y = mNormalizeROIs(1, j + n*mNbProposals);
                    const Float_T w = mNormalizeROIs(2, j + n*mNbProposals);
                    const Float_T h = mNormalizeROIs(3, j + n*mNbProposals);

                    const Float_T interLeft = std::max(x0, x);
                    const Float_T interRight = std::min(x0 + w0, x + w);
                    const Float_T interTop = std::max(y0, y);
                    const Float_T interBottom = std::min(y0 + h0, y + h);

                    if (interLeft < interRight && interTop < interBottom) {
                        const Float_T interArea = (interRight - interLeft)
                                                    * (interBottom - interTop);
                        const Float_T unionArea = w0 * h0 + w * h - interArea;
                        const Float_T IoU = interArea / unionArea;

                        if (IoU > mNMS_IoU_Threshold) {

                            // Suppress ROI
                            //ROIs[n].erase(ROIs[n].begin() + j);
                            mNormalizeROIs(0, j + n*mNbProposals) = 0.0;
                            mNormalizeROIs(1, j + n*mNbProposals) = 0.0;
                            mNormalizeROIs(2, j + n*mNbProposals) = 0.0;
                            mNormalizeROIs(3, j + n*mNbProposals) = 0.0;
                            continue;
                        }
                    }
                    ++j;
                }
            }
        }

        mNormalizeROIs.synchronizeHToD();
    }

    cudaSToOutputROIs(  mNbProposals, 
                        mNormalizeROIs.getDevicePtr(),
                        mOutputs.getDevicePtr(),
                        nbThread,
                        nbBlocks);

    Cell_Frame_CUDA::propagate();
    
    mDiffInputs.clearValid();
}

void N2D2::ProposalCell_Frame_CUDA::backPropagate()
{
    // No backpropagation for this layer
}

void N2D2::ProposalCell_Frame_CUDA::update()
{
    // Nothing to update
}

void N2D2::ProposalCell_Frame_CUDA::setOutputsSize()
{
    ProposalCell::setOutputsSize();

    if (mOutputs.empty()) {
        mOutputs.resize(mOutputsWidth,
                        mOutputsHeight,
                        mNbOutputs,
                        mInputs.dimB());
        mDiffInputs.resize(mOutputsWidth,
                           mOutputsHeight,
                           mNbOutputs,
                           mInputs.dimB());
    }
}

#endif