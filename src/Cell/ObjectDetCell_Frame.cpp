/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#include "Cell/ObjectDetCell_Frame.hpp"
#include "StimuliProvider.hpp"

N2D2::Registrar<N2D2::ObjectDetCell>
N2D2::ObjectDetCell_Frame::mRegistrar("Frame", N2D2::ObjectDetCell_Frame::create);

N2D2::ObjectDetCell_Frame::ObjectDetCell_Frame(const std::string& name,
                                                StimuliProvider& sp,
                                                const unsigned int nbOutputs,
                                                unsigned int nbAnchors,
                                                unsigned int nbProposals,
                                                unsigned int nbClass,
                                                Float_T nmsThreshold,
                                                std::vector<Float_T> scoreThreshold,
                                                std::vector<unsigned int> numParts,
                                                std::vector<unsigned int> numTemplates,
                                                const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors)
    : Cell(name, nbOutputs),
      ObjectDetCell(name, sp, nbOutputs, nbAnchors, nbProposals, nbClass, nmsThreshold, scoreThreshold, numParts, numTemplates, anchors),
      Cell_Frame<Float_T>(name, nbOutputs),
      mAnchors(anchors)
{
    // ctor
}

std::vector<N2D2::Float_T> N2D2::ObjectDetCell_Frame::getAnchor(unsigned int idx) const
{
    std::vector<Float_T> vect_anchor;
    vect_anchor.push_back(mAnchors[idx].x0);
    vect_anchor.push_back(mAnchors[idx].y0);
    vect_anchor.push_back(mAnchors[idx].x1);
    vect_anchor.push_back(mAnchors[idx].y1);
    return vect_anchor;
}

void N2D2::ObjectDetCell_Frame::initialize()
{
    if(mFeatureMapWidth == 0)
        mFeatureMapWidth = mStimuliProvider.getSizeX();

    if(mFeatureMapHeight == 0)
        mFeatureMapHeight = mStimuliProvider.getSizeY();

    if(mInputs.size() == 3)
    {
        if(mNumParts.size() != mNbClass)
            throw std::runtime_error("Specified NumParts must "
                                        "have the same size than NbClass in "
                                        " ProposalCell::Frame_CUDA " + mName);
        mMaxParts = *std::max_element(mNumParts.begin(),
                                        mNumParts.end());

        if(mNumTemplates.size() != mNbClass)
            throw std::runtime_error("Specified mNumTemplates must have"
                                        " the same size than NbClass in "
                                        " ProposalCell::Frame_CUDA " + mName);

        mMaxTemplates = *std::max_element(mNumTemplates.begin(),
                                            mNumTemplates.end());

    }
}

void N2D2::ObjectDetCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDToH();
    const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[0]);
    const Tensor<Float_T>& input_templates = mMaxTemplates > 0 ? tensor_cast<Float_T>(mInputs[1]) : Tensor<Float_T>();
    const Tensor<Float_T>& input_parts = mMaxParts > 0 ? tensor_cast<Float_T>(mInputs[2]) : Tensor<Float_T>();

    const unsigned int inputBatch = mOutputs.dimB()/(mNbProposals*mNbClass);
    const unsigned int offset = mNbAnchors*mNbClass;
    const double xRatio = std::ceil(mFeatureMapWidth / mInputs[0].dimX());
    const double yRatio = std::ceil(mFeatureMapHeight / mInputs[0].dimY());
    const float xOutputRatio = mStimuliProvider.getSizeX() / (float) mFeatureMapWidth;
    const float yOutputRatio = mStimuliProvider.getSizeY() / (float) mFeatureMapHeight;

    for(unsigned int batchPos = 0; batchPos < inputBatch; ++batchPos)
    {
        std::vector< std::vector<std::pair< Tensor<int>::Index, Float_T> >> ROIs;
        std::vector< std::vector<BBox_T >> ROIsPredicted;

        ROIs.resize(mNbClass);
        ROIsPredicted.resize(mNbClass);
        for(unsigned int cls = 0; cls < mNbClass; ++ cls)
        {
            //Keep ROIs with scores superior to the class threshold
            for (unsigned int anchor = 0; anchor < mNbAnchors; ++anchor)
            {
                for (unsigned int y = 0; y < input.dimY(); ++y) {
                    for (unsigned int x = 0; x < input.dimX(); ++x) {

                        const Float_T value = input( x,
                                                     y,
                                                     anchor + cls*mNbAnchors,
                                                     batchPos);

                        if(value >= mScoreThreshold[cls] && inference)
                        {
                            ROIs[cls].push_back(std::make_pair(Tensor<int>::Index(x, y, anchor, batchPos), value));
                        }
                        else if(value >= 0.0 && !inference)
                        {
                            ROIs[cls].push_back(std::make_pair(Tensor<int>::Index(x, y, anchor, batchPos),
                                                                value));
                        }
                    }
                }
            }

            //Sort ROIs highest to lowest score
            std::sort(ROIs[cls].begin(),
                      ROIs[cls].end(),
                      Utils::PairSecondPred<Tensor<int>::Index, Float_T,
                        std::greater<Float_T> >());

            if(inference)
            {
                std::vector<BBox_T> final_rois;
                BBox_T next_candidate;
                //Apply efficient Non Maximal Suppression
                std::reverse(ROIs[cls].begin(),ROIs[cls].end());
                while (final_rois.size() < mNbProposals && !ROIs[cls].empty()) {
                    next_candidate.x = ROIs[cls].back().first[0];
                    next_candidate.y = ROIs[cls].back().first[1];
                    next_candidate.w = ROIs[cls].back().first[2];
                    next_candidate.h = ROIs[cls].back().first[3];
                    next_candidate.s = ROIs[cls].back().second;
                    ROIs[cls].pop_back();

                    bool should_select = true;
                    const float x0 = input(   next_candidate.x, 
                                            next_candidate.y, 
                                            next_candidate.w + cls*mNbAnchors + offset, 
                                            next_candidate.h);
                    const float y0 = input(   next_candidate.x, 
                                            next_candidate.y, 
                                            next_candidate.w + cls*mNbAnchors + 2*offset, 
                                            next_candidate.h);
                    const float w0 = input(   next_candidate.x, 
                                            next_candidate.y, 
                                            next_candidate.w + cls*mNbAnchors + 3*offset, 
                                            next_candidate.h);
                    const float h0 = input(   next_candidate.x, 
                                            next_candidate.y, 
                                            next_candidate.w + cls*mNbAnchors + 4*offset, 
                                            next_candidate.h);

                    for (int j = static_cast<int>(final_rois.size()) - 1; j >= 0; --j) {

                        const float x = input(final_rois[j].x, 
                                            final_rois[j].y, 
                                            final_rois[j].w + cls*mNbAnchors + offset, 
                                            final_rois[j].h);

                        const float y = input(final_rois[j].x, 
                                            final_rois[j].y, 
                                            final_rois[j].w + cls*mNbAnchors + 2*offset, 
                                            final_rois[j].h);
                        const float w = input(final_rois[j].x, 
                                            final_rois[j].y, 
                                            final_rois[j].w + cls*mNbAnchors + 3*offset, 
                                            final_rois[j].h);
                        const float h = input(final_rois[j].x, 
                                            final_rois[j].y, 
                                            final_rois[j].w + cls*mNbAnchors + 4*offset, 
                                            final_rois[j].h);

                        const float interLeft = std::max(x0, x);
                        const float interRight = std::min(x0 + w0, x + w);
                        const float interTop = std::max(y0, y);
                        const float interBottom = std::min(y0 + h0, y + h);

                        if (interLeft < interRight && interTop < interBottom) {
                            const float interArea = (interRight - interLeft)
                                                        * (interBottom - interTop);
                            const float unionArea = w0 * h0 + w * h - interArea;
                            const float IoU = interArea / unionArea;

                            if (IoU > mNMS_IoU_Threshold) {
                                should_select = false;
                                break;

                            }
                        }

                    }
                    if (should_select) 
                        final_rois.push_back(next_candidate);
                }


                ROIsPredicted[cls].resize(final_rois.size());

                for(unsigned int proposal = 0; proposal < final_rois.size(); ++ proposal )
                {
                    ROIsPredicted[cls][proposal].x = final_rois[proposal].x;
                    ROIsPredicted[cls][proposal].y = final_rois[proposal].y;
                    ROIsPredicted[cls][proposal].w = final_rois[proposal].w;
                    ROIsPredicted[cls][proposal].h = final_rois[proposal].h;
                    ROIsPredicted[cls][proposal].s = final_rois[proposal].s;
                }
            }
            else
            {
                for(unsigned int proposal = 0; proposal < mNbProposals; ++ proposal )
                {
                    BBox_T final_rois;
                    final_rois.x = ROIs[cls][proposal].first[0];
                    final_rois.y = ROIs[cls][proposal].first[1];
                    final_rois.w = ROIs[cls][proposal].first[2];
                    final_rois.h = ROIs[cls][proposal].first[3];
                    final_rois.s = ROIs[cls][proposal].second;

                    ROIsPredicted[cls].push_back(final_rois);
                }

            }

        }

        for (unsigned int cls = 0; cls < mNbClass; ++cls)
        {

            for(unsigned int i = 0; i < mNbProposals; ++i)
            {

                const Float_T xbbEst = i < ROIsPredicted[cls].size() ? input(   ROIsPredicted[cls][i].x, 
                                                                                ROIsPredicted[cls][i].y, 
                                                                                ROIsPredicted[cls][i].w + cls*mNbAnchors + offset, 
                                                                                ROIsPredicted[cls][i].h) : 0.0;
                const Float_T ybbEst = i < ROIsPredicted[cls].size() ? input(   ROIsPredicted[cls][i].x, 
                                                                                ROIsPredicted[cls][i].y, 
                                                                                ROIsPredicted[cls][i].w + cls*mNbAnchors + 2*offset, 
                                                                                ROIsPredicted[cls][i].h) : 0.0;
                const Float_T wbbEst = i < ROIsPredicted[cls].size() ? input(   ROIsPredicted[cls][i].x, 
                                                                                ROIsPredicted[cls][i].y, 
                                                                                ROIsPredicted[cls][i].w + cls*mNbAnchors + 3*offset, 
                                                                                ROIsPredicted[cls][i].h) : 0.0;
                const Float_T hbbEst = i < ROIsPredicted[cls].size() ? input(   ROIsPredicted[cls][i].x, 
                                                                                ROIsPredicted[cls][i].y, 
                                                                                ROIsPredicted[cls][i].w + cls*mNbAnchors + 4*offset, 
                                                                                ROIsPredicted[cls][i].h) : 0.0;
                const Float_T score = i < ROIsPredicted[cls].size() ? ROIsPredicted[cls][i].s : 0.0;


                const unsigned int n = i + cls*mNbProposals 
                                            + batchPos*mNbProposals*mNbClass;

                mOutputs(0, n) = xbbEst;
                mOutputs(1, n) = ybbEst;
                mOutputs(2, n) = wbbEst;
                mOutputs(3, n) = hbbEst;
                mOutputs(4, n) = score;
                mOutputs(5, n) = (float) cls;

                if(mNumParts.size() > 0)
                {
                    for(unsigned int part = 0; part < mMaxParts; ++part)
                    {
                        if(part < mNumParts[cls] && i < ROIsPredicted[cls].size())
                        {
                                const int xa = ROIsPredicted[cls][i].x;
                                const int ya = ROIsPredicted[cls][i].y;
                                const int k = ROIsPredicted[cls][i].w;
                                const int b = ROIsPredicted[cls][i].h;

                                const Float_T partY = input_parts(xa,
                                                                    ya,
                                                                    k *mNumParts[cls]*2 + part*2 + 0
                                                                    + std::accumulate(mNumParts.begin(), mNumParts.begin() + cls, 0) * 2 * mNbAnchors,
                                                                    b);

                                const Float_T partX = input_parts(xa,
                                                                    ya,
                                                                    k *mNumParts[cls]*2 + part*2 + 1
                                                                    + std::accumulate(mNumParts.begin(), mNumParts.begin() + cls, 0) * 2 * mNbAnchors,
                                                                    b);

                                const int xa0 = (int)(mAnchors[k].x0 + xa * xRatio);
                                const int ya0 = (int)(mAnchors[k].y0 + ya * yRatio);
                                const int xa1 = (int)(mAnchors[k].x1 + xa * xRatio);
                                const int ya1 = (int)(mAnchors[k].y1 + ya * yRatio);

                                // Anchors width and height
                                const int wa = xa1 - xa0;
                                const int ha = ya1 - ya0;

                                // Anchor center coordinates (xac, yac)
                                const Float_T xac = xa0 + wa / 2.0;
                                const Float_T yac = ya0 + ha / 2.0;
                                const Float_T predPartY = ((partY) * ha + yac)*yOutputRatio ;
                                const Float_T predPartX = ((partX) * wa + xac)*xOutputRatio ;

                                mOutputs(6 + part*2 + 0, n) = predPartY;
                                mOutputs(6 + part*2 + 1, n) = predPartX;
                        }
                        else
                        {
                                mOutputs(6 + part*2 + 0, n) = 0.0;
                                mOutputs(6 + part*2 + 1, n) = 0.0;

                        }
                    }
                }
                if(mNumTemplates.size() > 0)
                {
                    for(unsigned int tplt = 0; tplt < mMaxTemplates; ++tplt)
                    {
                        if(tplt < mNumTemplates[cls] && i < ROIsPredicted[cls].size())
                        {
                                const int xa = ROIsPredicted[cls][i].x;
                                const int ya = ROIsPredicted[cls][i].y;
                                const int k = ROIsPredicted[cls][i].w;
                                const int b = ROIsPredicted[cls][i].h;

                                const Float_T templateY = input_templates(  xa,
                                                                            ya,
                                                                            k *mNumTemplates[cls]*3 + tplt*3 + 0
                                                                            + std::accumulate(mNumTemplates.begin(), mNumTemplates.begin() + cls, 0) * 3 * mNbAnchors,
                                                                            b);

                                const Float_T templateX = input_templates(  xa,
                                                                            ya,
                                                                            k *mNumTemplates[cls]*3 + tplt*3 + 1
                                                                            + std::accumulate(mNumTemplates.begin(), mNumTemplates.begin() + cls, 0) * 3 * mNbAnchors,
                                                                            b);
                                const Float_T templateZ = input_templates(  xa,
                                                                            ya,
                                                                            k *mNumTemplates[cls]*3 + tplt*3 + 2
                                                                            + std::accumulate(mNumTemplates.begin(), mNumTemplates.begin() + cls, 0) * 3 * mNbAnchors,
                                                                            b);

                                mOutputs(6 + mMaxParts*2 + tplt*3 + 0, n) = std::exp(templateY);
                                mOutputs(6 + mMaxParts*2 + tplt*3 + 1, n) = std::exp(templateX);
                                mOutputs(6 + mMaxParts*2 + tplt*3 + 2, n) = std::exp(templateZ);
                        }
                        else
                        {
                                mOutputs(6 + mMaxParts*2 + tplt*3 + 0, n) = 0.0;
                                mOutputs(6 + mMaxParts*2 + tplt*3 + 1, n) = 0.0;
                                mOutputs(6 + mMaxParts*2 + tplt*3 + 2, n) = 0.0;;

                        }
                    }
                }

            }
        }        
    }

    Cell_Frame<Float_T>::propagate();

    mDiffInputs.clearValid();
    
}

void N2D2::ObjectDetCell_Frame::backPropagate()
{
    // No backpropagation for this layer
}

void N2D2::ObjectDetCell_Frame::update()
{
    // Nothing to update
}

void N2D2::ObjectDetCell_Frame::setOutputsDims()
{
    ObjectDetCell::setOutputsDims();

    if (mOutputs.empty()) {
        mOutputs.resize({mOutputsDims[0],
                        mOutputsDims[1],
                        getNbOutputs(),
                        mNbProposals*mNbClass* mInputs.dimB()});
        mDiffInputs.resize({mOutputsDims[0],
                           mOutputsDims[1],
                           getNbOutputs(),
                           mNbProposals *mNbClass* mInputs.dimB()});
    }
}