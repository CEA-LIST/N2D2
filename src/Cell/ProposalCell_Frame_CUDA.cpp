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
                                            const unsigned int nbOutputs,
                                            unsigned int nbProposals,
                                            unsigned int scoreIndex,
                                            unsigned int IoUIndex,
                                            bool isNms,
                                            std::vector<double> meansFactor,
                                            std::vector<double> stdFactor,
                                            std::vector<unsigned int> numParts,
                                            std::vector<unsigned int> numTemplates)
    : Cell(name, nbOutputs),
      ProposalCell(name, sp, nbOutputs, nbProposals, scoreIndex, IoUIndex, isNms, meansFactor, stdFactor, numParts, numTemplates),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
}

void N2D2::ProposalCell_Frame_CUDA::initialize()
{
    //const unsigned int inputBatch = mOutputs.dimB()/mNbProposals;

    mMeansCUDA.resize(4);
    mStdCUDA.resize(4);
    for(unsigned int i = 0 ; i < 4; ++i)
    {
        mMeansCUDA(i) = mMeanFactor[i];
        mStdCUDA(i) = mStdFactor[i];
    }

    mNbClass = mInputs[2].dimX()*mInputs[2].dimY()*mInputs[2].dimZ();
    mNormalizeROIs.resize(1, 4, (mNbClass - mScoreIndex), mOutputs.dimB());
    mMaxCls.resize(1, 1, 1, mOutputs.dimB());

    mMaxCls.synchronizeHToD();    
    mNormalizeROIs.synchronizeHToD();    
    mMeansCUDA.synchronizeHToD();    
    mStdCUDA.synchronizeHToD();    

    if(mInputs.size() > 3)
    {
        mKeepIndex.resize(mOutputs.dimB()*2);
        mKeepIndex.fill(-1);
        if(mInputs.size() < 5)
        {
            if(mNumTemplates.size() != mNbClass)
                throw std::runtime_error("Specified mNumTemplates must have"
                                         " the same size than NbClass in "
                                         " ProposalCell::Frame_CUDA " + mName);

            mMaxTemplates = *std::max_element(mNumTemplates.begin(), 
                                              mNumTemplates.end());

            //Templates predictions need 3 output per detection 
            mTemplatesPrediction.resize(3, 
                                        //std::accumulate( mNumTemplates.begin(), mNumTemplates.end(), 0), 
                                        mMaxTemplates,
                                        mNbClass, 
                                        mOutputs.dimB());
            std::cout << "Layer ProposalCell::Frame_CUDA: "
                << mName << ": Provide templates prediction" << std::endl;

            mNumTemplatesPerClass.resize(mNbClass);
            for(unsigned int i = 0 ; i < mNbClass; ++i)
                mNumTemplatesPerClass(i) = mNumTemplates[i];

            mNumTemplatesPerClass.synchronizeHToD();
            mTemplatesPrediction.synchronizeHToD();

        }
        else if(mInputs.size() < 6)
        {
            if(mNumParts.size() != mNbClass)
                throw std::runtime_error("Specified NumParts must "
                                         "have the same size than NbClass in "
                                         " ProposalCell::Frame_CUDA " + mName);
            mMaxParts = *std::max_element(mNumParts.begin(), 
                                          mNumParts.end());

            //Parts predictions need 2 output per detection 
            mPartsPrediction.resize(2,
                                    //std::accumulate( mNumParts.begin(), mNumParts.end(), 0), 
                                    mMaxParts,
                                    mNbClass, 
                                    mOutputs.dimB());

            //Parts predictions visibility need 1 output per detection 
            mPartsVisibilityPrediction.resize(1, 
                                              //std::accumulate( mNumParts.begin(), mNumParts.end(), 0), 
                                              mMaxParts,
                                              mNbClass, 
                                              mOutputs.dimB());
            std::cout << "Layer ProposalCell::Frame_CUDA: "
                << mName << ": Provide parts prediction" << std::endl;

            mNumPartsPerClass.resize(mNbClass);
            for(unsigned int i = 0 ; i < mNbClass; ++i)
                mNumPartsPerClass(i) = mNumParts[i];

            mNumPartsPerClass.synchronizeHToD();
            mPartsPrediction.synchronizeHToD();
        }
        else if (mInputs.size() == 6)
        {
            if(mNumParts.size() != mNbClass)
                throw std::runtime_error("Specified NumParts must "
                                         "have the same size than NbClass in "
                                         " ProposalCell::Frame_CUDA " + mName);

            mMaxParts = *std::max_element(mNumParts.begin(), 
                                          mNumParts.end());

            mMaxTemplates = *std::max_element(mNumTemplates.begin(), 
                                              mNumTemplates.end());

            //Parts predictions need 2 output per detection 
            mPartsPrediction.resize(2,
                                    //std::accumulate( mNumParts.begin(), mNumParts.end(), 0), 
                                    mMaxParts,
                                    mNbClass, 
                                    mOutputs.dimB());

            //Parts predictions visibility need 1 output per detection 
            mPartsVisibilityPrediction.resize(1,
                                            //std::accumulate( mNumParts.begin(), mNumParts.end(), 0), 
                                              mMaxParts,
                                              mNbClass, 
                                              mOutputs.dimB());

            if(mNumTemplates.size() != mNbClass)
                throw std::runtime_error("Specified mNumTemplates must have"
                                         " the same size than NbClass in "
                                         " ProposalCell::Frame_CUDA " + mName);

            //Templates predictions need 3 output per detection 
            mTemplatesPrediction.resize(3, 
                                        //std::accumulate( mNumTemplates.begin(), mNumTemplates.end(), 0), 
                                        mMaxTemplates,
                                        mNbClass, 
                                        mOutputs.dimB());
            std::cout << "Layer ProposalCell::Frame_CUDA: "
                << mName << ": Provide parts and templates prediction" << std::endl;

            mNumPartsPerClass.resize(mNbClass);
            mNumTemplatesPerClass.resize(mNbClass);
            for(unsigned int i = 0 ; i < mNbClass; ++i)
            {
                mNumPartsPerClass(i) = mNumParts[i];
                mNumTemplatesPerClass(i) = mNumTemplates[i];
            }
            mNumPartsPerClass.synchronizeHToD();
            mNumTemplatesPerClass.synchronizeHToD();
            mPartsPrediction.synchronizeHToD();
            mTemplatesPrediction.synchronizeHToD();

        }
        else
            throw std::runtime_error("Too much entry for ProposalCell::Frame_CUDA: " + mName);      

        //mKeepIndex.synchronizeHToD();
    }

    std::cout << "PropocalCell::Frame " << mName << " provide " 
            <<  mNbClass << " class\n"
            << std::endl;
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
                        mNbClass,

                        mMaxParts,
                        mMaxTemplates,
                        mKeepMax,
                        mMaxParts > 0 ? true : false,
                        mMaxTemplates > 0 ? true : false,
                        
                        normX,
                        normY,
                        mMeansCUDA.getDevicePtr(),
                        mStdCUDA.getDevicePtr(),
                        mNumPartsPerClass.getDevicePtr(),
                        mNumTemplatesPerClass.getDevicePtr(),
                        mInputs[0].getDevicePtr(), 
                        mInputs[1].getDevicePtr(),
                        mInputs[2].getDevicePtr(),
                        (mMaxParts > 0 || mMaxTemplates > 0)  ? mInputs[3].getDevicePtr() 
                        : mInputs[2].getDevicePtr(),
                        mMaxParts > 0 ? mInputs[4].getDevicePtr() 
                        : mInputs[2].getDevicePtr(),
                        (mMaxParts > 0 && mMaxTemplates > 0) ? mInputs[5].getDevicePtr()
                        : mInputs[2].getDevicePtr(),
                        mNormalizeROIs.getDevicePtr(),
                        mMaxCls.getDevicePtr(),
                        mPartsPrediction.getDevicePtr(),
                        mTemplatesPrediction.getDevicePtr(),
                        mScoreThreshold,
                        nbThread,
                        nbBlocks);



    if(mApplyNMS)
    {
        mMaxCls.synchronizeDToH();
        mNormalizeROIs.synchronizeDToH();

        for(unsigned int n = 0; n < inputBatch; ++n)
        {
            // Non-Maximum Suppression (NMS)
            for(unsigned int cls = 0; cls < (mNbClass - mScoreIndex) ; ++cls)
            {
                for (unsigned int i = 0; i < mNbProposals - 1;
                    ++i)
                {
                    const Float_T x0 = mNormalizeROIs(0, 0, cls, i + n*mNbProposals);
                    const Float_T y0 = mNormalizeROIs(0, 1, cls, i + n*mNbProposals);
                    const Float_T w0 = mNormalizeROIs(0, 2, cls, i + n*mNbProposals);
                    const Float_T h0 = mNormalizeROIs(0, 3, cls, i + n*mNbProposals);

                    for (unsigned int j = i + 1; j < mNbProposals; ) {

                        const Float_T x = mNormalizeROIs(0, 0, cls, j + n*mNbProposals);
                        const Float_T y = mNormalizeROIs(0, 1, cls, j + n*mNbProposals);
                        const Float_T w = mNormalizeROIs(0, 2, cls, j + n*mNbProposals);
                        const Float_T h = mNormalizeROIs(0, 3, cls, j + n*mNbProposals);

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
                                mNormalizeROIs(0, 0, cls, j + n*mNbProposals) = 0.0;
                                mNormalizeROIs(0, 1, cls, j + n*mNbProposals) = 0.0;
                                mNormalizeROIs(0, 2, cls, j + n*mNbProposals) = 0.0;
                                mNormalizeROIs(0, 3, cls, j + n*mNbProposals) = 0.0;
                                continue;
                            }
                        }
                        ++j;
                    }
                }
            }
            unsigned int out = 0;
            
            for(unsigned int cls = 0; cls < (mNbClass - mScoreIndex) 
                    && out < mNbProposals; ++cls)
            {
                for (unsigned int i = 0; i < mNbProposals && out < mNbProposals ;
                    ++i)
                {
                    //Read before erase and write
                    const Float_T x = mNormalizeROIs(0, 0, cls, i + n*mNbProposals);
                    const Float_T y = mNormalizeROIs(0, 1, cls, i + n*mNbProposals);
                    const Float_T w = mNormalizeROIs(0, 2, cls, i + n*mNbProposals);
                    const Float_T h = mNormalizeROIs(0, 3, cls, i + n*mNbProposals);

                    
                    if(w > 0.0 && h > 0.0)
                    {
                        //Erase before write
                        mNormalizeROIs(0, 0, cls, i + n*mNbProposals) = 0.0;
                        mNormalizeROIs(0, 1, cls, i + n*mNbProposals) = 0.0;
                        mNormalizeROIs(0, 2, cls, i + n*mNbProposals) = 0.0;
                        mNormalizeROIs(0, 3, cls, i + n*mNbProposals) = 0.0;
                        
                        //Write result
                        mNormalizeROIs(0, 0, 0, out + n*mNbProposals) = x;
                        mNormalizeROIs(0, 1, 0, out + n*mNbProposals) = y;
                        mNormalizeROIs(0, 2, 0, out + n*mNbProposals) = w;
                        mNormalizeROIs(0, 3, 0, out + n*mNbProposals) = h;
                        mMaxCls(out + n*mNbProposals) = (int) (cls + mScoreIndex);

                        mKeepIndex(out*2 + 0) = i;
                        mKeepIndex(out*2 + 1) = cls + mScoreIndex;
                        ++out;
                    }
                }
            }
        }
        mNormalizeROIs.synchronizeHToD();
        mMaxCls.synchronizeHToD();
        mKeepIndex.synchronizeHToD();
    }

    cudaSToOutputROIs(  mNbProposals, 
                        mScoreIndex,
                        mNbClass,
                        mNbOutputs,
                        mMaxParts,
                        mMaxTemplates,
                        mMaxParts > 0 ? true: false,
                        mMaxTemplates > 0 ? true: false,
                        mNumPartsPerClass.getDevicePtr(),
                        mNumTemplatesPerClass.getDevicePtr(),
                        mMaxCls.getDevicePtr(),
                        mNormalizeROIs.getDevicePtr(),
                        mKeepIndex.getDevicePtr(),
                        mPartsPrediction.getDevicePtr(),
                        mTemplatesPrediction.getDevicePtr(),
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