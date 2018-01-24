/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
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

#include "Cell/ProposalCell_Frame.hpp"

N2D2::Registrar<N2D2::ProposalCell>
N2D2::ProposalCell_Frame::mRegistrar("Frame", N2D2::ProposalCell_Frame::create);

N2D2::ProposalCell_Frame::ProposalCell_Frame(const std::string& name,
                                            StimuliProvider& sp,
                                            unsigned int nbProposals,
                                            unsigned int scoreIndex,
                                            unsigned int IoUIndex,
                                            bool isNms,
                                            std::vector<double> meansFactor,
                                            std::vector<double> stdFactor)
    : Cell(name, 4),
      ProposalCell(name, sp, nbProposals, scoreIndex, IoUIndex, isNms, meansFactor, stdFactor),
      Cell_Frame(name, 4)
{
    // ctor
}


void N2D2::ProposalCell_Frame::initialize()
{
    if (mInputs.size() < 3) {
        throw std::runtime_error("At least three inputs are required for"
                                 " ProposalCell " + mName);
    }

    if (mInputs[0].dimX() * mInputs[0].dimY() * mInputs[0].dimZ() != 4) {
        throw std::runtime_error("The first input (BBox Ref) must have a XYZ size of 4 for"
                                 " ProposalCell " + mName);
    }
    if((mInputs[1].dimX()*mInputs[1].dimY()*mInputs[1].dimZ()) 
        / (mInputs[2].dimX()*mInputs[2].dimY()*mInputs[2].dimZ()) != 4)
        throw std::runtime_error("The second input and the third inputs must have the same"
                                 " class number in ProposalCell " + mName);

    mNbClass = mInputs[2].dimX()*mInputs[2].dimY()*mInputs[2].dimZ();

    std::cout << "PropocalCell::Frame " << mName << " provide " 
            <<  mNbClass << " class" << std::endl;
}

void N2D2::ProposalCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDToH();

    const Float_T normX = 1.0 / (mStimuliProvider.getSizeX() - 1) ;
    const Float_T normY = 1.0 / (mStimuliProvider.getSizeY() - 1) ;
    const unsigned int inputBatch = mOutputs.dimB()/mNbProposals; 

    if(mKeepMax)
    {
        std::vector< std::vector<BBox_T> > ROIs;
        ROIs.resize(inputBatch);
        for(unsigned int n = 0; n < inputBatch; ++n)
        {
            ROIs[n].resize(mNbProposals);
            for (unsigned int proposal = 0; proposal < mNbProposals; ++proposal)
            {
                const unsigned int batchPos = proposal + n*mNbProposals;
                unsigned int cls = mScoreIndex;
                for(unsigned int i = mScoreIndex + 1; i < mNbClass; ++i)
                {
                    if(mInputs[2](i, batchPos) > mInputs[2](cls, batchPos))
                        cls = i;
                }

                const Float_T xbbRef = mInputs[0](0, batchPos)*normX;
                const Float_T ybbRef = mInputs[0](1, batchPos)*normY;
                const Float_T wbbRef = mInputs[0](2, batchPos)*normX;
                const Float_T hbbRef = mInputs[0](3, batchPos)*normY;

                const Float_T xbbEst = mInputs[1](0 + cls*4, batchPos)*mStdFactor[0] + mMeanFactor[0];
                const Float_T ybbEst = mInputs[1](1 + cls*4, batchPos)*mStdFactor[1] + mMeanFactor[1];
                const Float_T wbbEst = mInputs[1](2 + cls*4, batchPos)*mStdFactor[2] + mMeanFactor[2];
                const Float_T hbbEst = mInputs[1](3 + cls*4, batchPos)*mStdFactor[3] + mMeanFactor[3];

                Float_T x = xbbEst*wbbRef + xbbRef + wbbRef/2.0 
                                - (wbbRef/2.0)*std::exp(wbbEst);
                Float_T y = ybbEst*hbbRef + ybbRef + hbbRef/2.0 
                                - (hbbRef/2.0)*std::exp(hbbEst);
                Float_T w = wbbRef*std::exp(wbbEst);
                Float_T h = hbbRef*std::exp(hbbEst);

                /**Clip values**/
                if(x < 0.0)
                {
                    w += x;
                    x = 0.0;
                }

                if(y < 0.0)
                {
                    h += y;
                    y = 0.0;
                }

                w = ((w + x) > 1.0) ? (1.0 - x) / normX : w / normX;
                h = ((h + y) > 1.0) ? (1.0 - y) / normY : h / normY;

                x /= normX;
                y /= normY;

                if( mInputs[2](cls, batchPos) >= mScoreThreshold )
                {
                    ROIs[n][proposal] = BBox_T(x,y,w,h);
                }
                else
                {
                    ROIs[n][proposal].x = 0.0;
                    ROIs[n][proposal].y = 0.0;
                    ROIs[n][proposal].w = 0.0;
                    ROIs[n][proposal].h = 0.0;
                }
            }   

            for (unsigned int proposal = 0; proposal < mNbProposals; ++proposal)
            {
                const unsigned int batchPos = proposal + n*mNbProposals;

                mOutputs(0, batchPos) = ROIs[n][proposal].x;
                mOutputs(1, batchPos) = ROIs[n][proposal].y;
                mOutputs(2, batchPos) = ROIs[n][proposal].w;
                mOutputs(3, batchPos) = ROIs[n][proposal].h;   
            }
        }
    }
    else
    {
        std::vector< std::vector< std::vector<BBox_T> > > ROIs;
        ROIs.resize(inputBatch);
        for(unsigned int n = 0; n < inputBatch; ++n)
        {

            ROIs[n].resize(mNbClass);

            unsigned int nbRoiDetected = 0;

            for(unsigned int cls = mScoreIndex; cls < mNbClass; ++ cls)
            {
                for (unsigned int proposal = 0; proposal < mNbProposals; ++proposal)
                {
                    const unsigned int batchPos = proposal + n*mNbProposals;

                    const Float_T xbbRef = mInputs[0](0, batchPos)*normX;
                    const Float_T ybbRef = mInputs[0](1, batchPos)*normY;
                    const Float_T wbbRef = mInputs[0](2, batchPos)*normX;
                    const Float_T hbbRef = mInputs[0](3, batchPos)*normY;

                    const Float_T xbbEst = mInputs[1](0 + cls*4, batchPos)*mStdFactor[0] + mMeanFactor[0];
                    const Float_T ybbEst = mInputs[1](1 + cls*4, batchPos)*mStdFactor[1] + mMeanFactor[1];
                    const Float_T wbbEst = mInputs[1](2 + cls*4, batchPos)*mStdFactor[2] + mMeanFactor[2];
                    const Float_T hbbEst = mInputs[1](3 + cls*4, batchPos)*mStdFactor[3] + mMeanFactor[3];

                    Float_T x = xbbEst*wbbRef + xbbRef + wbbRef/2.0 
                                    - (wbbRef/2.0)*std::exp(wbbEst);
                    Float_T y = ybbEst*hbbRef + ybbRef + hbbRef/2.0 
                                    - (hbbRef/2.0)*std::exp(hbbEst);
                    Float_T w = wbbRef*std::exp(wbbEst);
                    Float_T h = hbbRef*std::exp(hbbEst);

                    /**Clip values**/
                    if(x < 0.0)
                    {
                        w += x;
                        x = 0.0;
                    }

                    if(y < 0.0)
                    {
                        h += y;
                        y = 0.0;
                    }

                    w = ((w + x) > 1.0) ? (1.0 - x) / normX : w / normX;
                    h = ((h + y) > 1.0) ? (1.0 - y) / normY : h / normY;

                    x /= normX;
                    y /= normY;

                    if( mInputs[2](cls, batchPos) >= mScoreThreshold )
                        ROIs[n][cls].push_back(BBox_T(x,y,w,h));
                }   

                if(mApplyNMS)
                {

                    if(ROIs[n][cls].size() > 0)
                    {
                        // Non-Maximum Suppression (NMS)
                        for (unsigned int i = 0; i < ROIs[n][cls].size() - 1;
                            ++i)
                        {
                            const Float_T x0 = ROIs[n][cls][i].x;
                            const Float_T y0 = ROIs[n][cls][i].y;
                            const Float_T w0 = ROIs[n][cls][i].w;
                            const Float_T h0 = ROIs[n][cls][i].h;

                            for (unsigned int j = i + 1; j < ROIs[n][cls].size(); ) {
                
                                const Float_T x = ROIs[n][cls][j].x;
                                const Float_T y = ROIs[n][cls][j].y;
                                const Float_T w = ROIs[n][cls][j].w;
                                const Float_T h = ROIs[n][cls][j].h;

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
                                        ROIs[n][cls].erase(ROIs[n][cls].begin() + j);
                                        continue;
                                    }
                                }
                                ++j;
                            }
                        }
                    }
                }
                nbRoiDetected += ROIs[n][cls].size();
            }

            unsigned int totalIdx = 0;
            //unsigned int cls = mScoreIndex;
            for (unsigned int cls = mScoreIndex; cls < mNbClass && totalIdx < mNbProposals; ++cls)
            {
                for(unsigned int i = 0; i < ROIs[n][cls].size() && totalIdx < mNbProposals; ++i)
                {
                    const unsigned int batchPos = totalIdx + n*mNbProposals;
                    mOutputs(0, batchPos) = ROIs[n][cls][i].x;
                    mOutputs(1, batchPos) = ROIs[n][cls][i].y;
                    mOutputs(2, batchPos) = ROIs[n][cls][i].w;
                    mOutputs(3, batchPos) = ROIs[n][cls][i].h;   

                    totalIdx++;
                }
            }

            for(unsigned int rest = totalIdx; rest < mNbProposals; ++rest)
            {
                    const unsigned int batchPos = rest + n*mNbProposals;
                    mOutputs(0, batchPos) = 0.0;
                    mOutputs(1, batchPos) = 0.0;
                    mOutputs(2, batchPos) = 0.0;
                    mOutputs(3, batchPos) = 0.0;   
            }
        }
    }

    mDiffInputs.clearValid();
}

void N2D2::ProposalCell_Frame::backPropagate()
{
    // No backpropagation for this layer
}

void N2D2::ProposalCell_Frame::update()
{
    // Nothing to update
}

void N2D2::ProposalCell_Frame::setOutputsSize()
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
