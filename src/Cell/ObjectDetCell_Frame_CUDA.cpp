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
#ifdef CUDA

#include "Cell/ObjectDetCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include <thrust/device_ptr.h>

N2D2::Registrar<N2D2::ObjectDetCell>
N2D2::ObjectDetCell_Frame_CUDA::mRegistrar("Frame_CUDA", N2D2::ObjectDetCell_Frame_CUDA::create);



N2D2::ObjectDetCell_Frame_CUDA::ObjectDetCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
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

    : Cell(deepNet, name, nbOutputs),
      ObjectDetCell(deepNet, name, sp, nbOutputs, nbAnchors, nbProposals, nbClass, nmsThreshold, scoreThreshold, numParts, numTemplates),
      Cell_Frame_CUDA<Float_T>(deepNet, name, nbOutputs),
      mAnchors(anchors)
{
    // ctor
}

std::vector<N2D2::Float_T> N2D2::ObjectDetCell_Frame_CUDA::getAnchor(unsigned int idx) const
{
    std::vector<Float_T> vect_anchor;
    vect_anchor.push_back(mAnchors[idx].x0);
    vect_anchor.push_back(mAnchors[idx].y0);
    vect_anchor.push_back(mAnchors[idx].x1);
    vect_anchor.push_back(mAnchors[idx].y1);
    return vect_anchor;
}

void N2D2::ObjectDetCell_Frame_CUDA::initialize()
{
    const unsigned int outputMaxSizePerCls = mNbAnchors*mInputs[0].dimY()*mInputs[0].dimX();
    const unsigned int nbBlocks = std::ceil(outputMaxSizePerCls/(float) 32.0);

    GPU_BLOCK_GRID.push_back(dim3(nbBlocks, mNbClass, (int) mInputs.dimB()));
    GPU_THREAD_GRID.push_back(dim3(32,1,1));

    mPixelMap.resize({mInputs[0].dimX(),
                      mInputs[0].dimY(),
                      mNbAnchors,
                      mNbClass,
                      mInputs.dimB()}, -1);

    mPixelMapSorted.resize({mInputs[0].dimX(),
                            mInputs[0].dimY(),
                            mNbAnchors,
                            mNbClass,
                            mInputs.dimB()}, -1);

    mPixelMap.synchronizeHToD();

    mThreshPerClass.resize({mNbClass}, 0.0);

    for(unsigned int i = 0; i < mNbClass; ++i)
        mThreshPerClass(i) = mScoreThreshold[i];

    mThreshPerClass.synchronizeHToD();
    if(mFeatureMapWidth == 0)
        mFeatureMapWidth = mStimuliProvider.getSizeX();

    if(mFeatureMapHeight == 0)
        mFeatureMapHeight = mStimuliProvider.getSizeY();

    mScores.resize({mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mNbAnchors,
                        mNbClass,
                        mInputs.dimB()});
    mScoresIndex.resize({mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mNbAnchors,
                        mNbClass,
                        mInputs.dimB()});
    mScoresFiltered.resize({mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        mNbAnchors,
                        mNbClass,
                        mInputs.dimB()});
    mScores.synchronizeHToD();
    mScoresIndex.synchronizeHToD();
    mScoresFiltered.synchronizeHToD();

    mROIsBBOxFinal.resize({5, mNbProposals, mInputs.dimB(), mNbClass});
    mROIsMapAnchorsFinal.resize({5, mNbProposals, mInputs.dimB(),  mNbClass});
    mROIsIndexFinal.resize({mInputs.dimB(),  mNbClass});

    mX_index.resize({mInputs[0].dimX() * mInputs[0].dimY() * mNbAnchors * mNbClass * mInputs[0].dimB()}, 0.0);
    mX_index.synchronizeHToD();
    mY_index.resize({mInputs[0].dimX() * mInputs[0].dimY() * mNbAnchors * mNbClass * mInputs[0].dimB()}, 0.0);
    mY_index.synchronizeHToD();
    mW_index.resize({mInputs[0].dimX() * mInputs[0].dimY() * mNbAnchors * mNbClass * mInputs[0].dimB()}, 0.0);
    mW_index.synchronizeHToD();
    mH_index.resize({mInputs[0].dimX() * mInputs[0].dimY() * mNbAnchors * mNbClass * mInputs[0].dimB()}, 0.0);
    mH_index.synchronizeHToD();

    if(mInputs.size() == 3)
    {
        if(mNumParts.size() != mNbClass)
            throw std::runtime_error("Specified NumParts must "
                                        "have the same size than NbClass in "
                                        " ProposalCell::Frame_CUDA " + mName);

        mMaxParts = *std::max_element(mNumParts.begin(), mNumParts.end());

        //Parts predictions need 2 output per detection
        mPartsPrediction.resize({2,
                                //std::accumulate( mNumParts.begin(), mNumParts.end(), 0),
                                mMaxParts,
                                mNbClass,
                                mOutputs.dimB()});


        if(mNumTemplates.size() != mNbClass)
            throw std::runtime_error("Specified mNumTemplates must have"
                                        " the same size than NbClass in "
                                        " ProposalCell::Frame_CUDA " + mName);

        mMaxTemplates = *std::max_element(mNumTemplates.begin(),
                                            mNumTemplates.end());

        //Templates predictions need 3 output per detection
        mTemplatesPrediction.resize({3,
                                    //std::accumulate( mNumTemplates.begin(), mNumTemplates.end(), 0),
                                    mMaxTemplates,
                                    mNbClass,
                                    mOutputs.dimB()});
        std::cout << "Layer ProposalCell::Frame_CUDA: "
            << mName << ": Provide parts and templates prediction" << std::endl;

        mNumPartsPerClass.resize({mNbClass});
        mNumTemplatesPerClass.resize({mNbClass});
        for(unsigned int i = 0 ; i < mNbClass; ++i)
        {
            mNumPartsPerClass(i) = mNumParts[i];
            mNumTemplatesPerClass(i) = mNumTemplates[i];
        }
        mNumPartsPerClass.synchronizeHToD();
        mNumTemplatesPerClass.synchronizeHToD();
        mPartsPrediction.synchronizeHToD();
        mTemplatesPrediction.synchronizeHToD();

        mGPUAnchors.resize({4,mNbAnchors*mNbClass});
        for(unsigned int i = 0; i< mNbAnchors*mNbClass; ++i)
        {
            mGPUAnchors(0, i) = mAnchors[i].x0;
            mGPUAnchors(1, i) = mAnchors[i].y0;
            mGPUAnchors(2, i) = mAnchors[i].x1;
            mGPUAnchors(3, i) = mAnchors[i].y1;
        }
        mGPUAnchors.synchronizeHToD();

    }

}

void N2D2::ObjectDetCell_Frame_CUDA::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();
    std::shared_ptr<CudaDeviceTensor<Float_T> > input_templates 
        = mNumTemplates.size() > 0 ? cuda_device_tensor_cast_nocopy<Float_T>(mInputs[1])  
                                    : std::shared_ptr<CudaDeviceTensor<Float_T> >();

    std::shared_ptr<CudaDeviceTensor<Float_T> > input_parts 
        = mNumParts.size() > 0 ? cuda_device_tensor_cast_nocopy<Float_T>(mInputs[2]) 
                                : std::shared_ptr<CudaDeviceTensor<Float_T> >();

    std::shared_ptr<CudaDeviceTensor<Float_T> > input0
            = cuda_device_tensor_cast_nocopy<Float_T>(mInputs[0]);

    const unsigned int inputBatchOffset = mInputs[0].dimX()*mInputs[0].dimY()*mInputs[0].dimZ();

    const double xRatio = std::ceil(mFeatureMapWidth / mInputs[0].dimX());
    const double yRatio = std::ceil(mFeatureMapHeight / mInputs[0].dimY());
    const float xOutputRatio = mStimuliProvider.getSizeX() / (float) mFeatureMapWidth;
    const float yOutputRatio = mStimuliProvider.getSizeY() / (float) mFeatureMapHeight;



    mPixelMap.synchronizeDToH();

    mPixelMap.assign({mInputs[0].dimX(),
                      mInputs[0].dimY(),
                      mNbAnchors,
                      mNbClass,
                      mInputs.dimB()}, -1);

    mPixelMapSorted.assign({mInputs[0].dimX(),
                            mInputs[0].dimY(),
                            mNbAnchors,
                            mNbClass,
                            mInputs.dimB()}, -1);


    mPixelMap.synchronizeHToD();
    mPixelMapSorted.synchronizeHToD();

    cudaSReduceIndex(  mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors,
                       inputBatchOffset,
                       mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors*mNbClass,
                       mThreshPerClass.getDevicePtr(),
                       input0->getDevicePtr(),
                       mPixelMap.getDevicePtr(),
                       mScores.getDevicePtr(),
                       GPU_BLOCK_GRID[0],
                       GPU_THREAD_GRID[0]);
    std::vector<std::vector <unsigned int> > count(mInputs.dimB(),
                                                   std::vector<unsigned int>(mNbClass));

    for(unsigned int batchPos = 0; batchPos < mInputs.dimB(); ++batchPos)
    {

        for(unsigned int cls = 0; cls < mNbClass; ++cls)
        {

            const int pixelOffset = cls*mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors 
                                        + batchPos*mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors*mNbClass;

            const int nbMapDet = copy_if_INT32( mPixelMap.getDevicePtr() + pixelOffset,
                                            mPixelMapSorted.getDevicePtr() + pixelOffset,
                                            mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors);

            const int nbScoreDet = copy_if_FP32(mScores.getDevicePtr() + pixelOffset,
                                                mScoresFiltered.getDevicePtr() + pixelOffset,
                                                mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors);

            if (nbScoreDet != nbMapDet)
                throw std::runtime_error(
                    "Dont find the same number of valid boxes");

            count[batchPos][cls] = nbMapDet;
        }
    }
    std::vector< std::vector< std::vector<BBox_T >>> ROIs(  mNbClass, 
                                                            std::vector< std::vector <BBox_T>>(mInputs.dimB()));

    std::vector< std::vector< std::vector<BBox_T >>> ROIsAnchors(   mNbClass, 
                                                                    std::vector< std::vector <BBox_T>>(mInputs.dimB()));

    for(unsigned int cls = 0; cls < mNbClass; ++cls)
    {
        const int offset = cls*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY();

        for(unsigned int batchPos = 0; batchPos < mInputs.dimB(); ++batchPos)
        {
            const int batchOffset = batchPos*inputBatchOffset;

            if(count[batchPos][cls] > 0)
            {
                const int offsetBase = mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY();

                const int offsetCpy = cls*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY()
                                        + batchPos*mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY();

                unsigned int nbElementNMS =  count[batchPos][cls];

                thrust_sort_keys_INT32(     mScoresFiltered.getDevicePtr() + offsetCpy,
                                            mPixelMapSorted.getDevicePtr() + offsetCpy,
                                            nbElementNMS,
                                            0);

                thrust_gather_INT32(mPixelMapSorted.getDevicePtr() + offsetCpy,
                                    input0->getDevicePtr() + offsetBase + offset + batchOffset,
                                    mX_index.getDevicePtr(),
                                    count[batchPos][cls],
                                    0,
                                    0);

                thrust_gather_INT32(mPixelMapSorted.getDevicePtr() + offsetCpy,
                                    input0->getDevicePtr() + 2*offsetBase + offset + batchOffset,
                                    mY_index.getDevicePtr(),
                                    count[batchPos][cls],
                                    0,
                                    0);

                thrust_gather_INT32(mPixelMapSorted.getDevicePtr() + offsetCpy,
                                    input0->getDevicePtr() + 3*offsetBase + offset + batchOffset,
                                    mW_index.getDevicePtr(),
                                    count[batchPos][cls],
                                    0,
                                    0);

                thrust_gather_INT32(mPixelMapSorted.getDevicePtr() + offsetCpy,
                                    input0->getDevicePtr() + 4*offsetBase + offset + batchOffset,
                                    mH_index.getDevicePtr(),
                                    count[batchPos][cls],
                                    0,
                                    0);

                mX_index.synchronizeDToH(0, nbElementNMS);
                mY_index.synchronizeDToH(0, nbElementNMS);
                mW_index.synchronizeDToH(0, nbElementNMS);
                mH_index.synchronizeDToH(0, nbElementNMS);

                mPixelMapSorted.synchronizeDToH(offsetCpy, nbElementNMS);
                mScoresFiltered.synchronizeDToH(offsetCpy, nbElementNMS);

                for(unsigned int idx = 0; idx < nbElementNMS; ++idx)
                {

                    ROIs[cls][batchPos].push_back(BBox_T(  mX_index(idx),
                                                           mY_index(idx),
                                                           mW_index(idx),
                                                           mH_index(idx),
                                                           mScoresFiltered(idx + offsetCpy)));

                    ROIsAnchors[cls][batchPos].push_back(BBox_T(  mPixelMapSorted(idx + offsetCpy)%mInputs[0].dimX(),
                                                                  (mPixelMapSorted(idx + offsetCpy)/mInputs[0].dimX())%mInputs[0].dimY(),
                                                                  (mPixelMapSorted(idx + offsetCpy)/(mInputs[0].dimX()*mInputs[0].dimY()))%mNbAnchors,
                                                                  0.0,
                                                                  0.0));
                }
                
                if(inference)
                {
                    std::vector<BBox_T> final_rois;
                    std::vector<BBox_T> final_anchors;

                    BBox_T next_candidate;
                    BBox_T next_anchors;
                    std::reverse(ROIs[cls][batchPos].begin(),ROIs[cls][batchPos].end());
                    std::reverse(ROIsAnchors[cls][batchPos].begin(),ROIsAnchors[cls][batchPos].end());

                    while (final_rois.size() < mNbProposals && !ROIs[cls][batchPos].empty()) {
                        next_candidate = ROIs[cls][batchPos].back();
                        ROIs[cls][batchPos].pop_back();
                        next_anchors = ROIsAnchors[cls][batchPos].back();
                        ROIsAnchors[cls][batchPos].pop_back();

                        bool should_select = true;
                        const float x0 = next_candidate.x;
                        const float y0 = next_candidate.y;
                        const float w0 = next_candidate.w;
                        const float h0 = next_candidate.h;

                        for (int j = static_cast<int>(final_rois.size()) - 1; j >= 0; --j) {

                            const float x = final_rois[j].x;
                            const float y = final_rois[j].y;
                            const float w = final_rois[j].w;
                            const float h = final_rois[j].h;
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

                        if (should_select) {
                            final_rois.push_back(next_candidate);
                            final_anchors.push_back(next_anchors);
                        }
                    }

                    ROIs[cls][batchPos].resize(final_rois.size());
                    ROIsAnchors[cls][batchPos].resize(final_anchors.size());

                    for(unsigned int proposal = 0; proposal < final_rois.size(); ++ proposal )
                    {
                        mROIsBBOxFinal({0, proposal, batchPos, cls}) = final_rois[proposal].x;
                        mROIsBBOxFinal({1, proposal, batchPos, cls}) = final_rois[proposal].y;
                        mROIsBBOxFinal({2, proposal, batchPos, cls}) = final_rois[proposal].w;
                        mROIsBBOxFinal({3, proposal, batchPos, cls}) = final_rois[proposal].h;
                        mROIsBBOxFinal({4, proposal, batchPos, cls}) = final_rois[proposal].s;
                        mROIsMapAnchorsFinal({0, proposal, batchPos, cls}) = final_anchors[proposal].x;
                        mROIsMapAnchorsFinal({1, proposal, batchPos, cls}) = final_anchors[proposal].y;
                        mROIsMapAnchorsFinal({2, proposal, batchPos, cls}) = final_anchors[proposal].w;
                        mROIsMapAnchorsFinal({3, proposal, batchPos, cls}) = final_anchors[proposal].h;
                        mROIsMapAnchorsFinal({4, proposal, batchPos, cls}) = final_anchors[proposal].s;
                    }
                    for(unsigned int proposal = final_rois.size(); proposal < mNbProposals; ++ proposal )
                    {
                        mROIsBBOxFinal({0, proposal, batchPos, cls}) = 0.0;
                        mROIsBBOxFinal({1, proposal, batchPos, cls}) = 0.0;
                        mROIsBBOxFinal({2, proposal, batchPos, cls}) = 0.0;
                        mROIsBBOxFinal({3, proposal, batchPos, cls}) = 0.0;
                        mROIsBBOxFinal({4, proposal, batchPos, cls}) = 0.0;
                        mROIsMapAnchorsFinal({0, proposal, batchPos, cls}) = 0.0;
                        mROIsMapAnchorsFinal({1, proposal, batchPos, cls}) = 0.0;
                        mROIsMapAnchorsFinal({2, proposal, batchPos, cls}) = 0.0;
                        mROIsMapAnchorsFinal({3, proposal, batchPos, cls}) = 0.0;
                        mROIsMapAnchorsFinal({4, proposal, batchPos, cls}) = 0.0;
                    }

                    mROIsIndexFinal(batchPos, cls) = final_rois.size();
                }
                else
                {
                    for(unsigned int proposal = 0; proposal < mNbProposals; ++ proposal )
                    {
                        mROIsBBOxFinal({0, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIs[cls][batchPos][proposal].x : 0.0;
                        mROIsBBOxFinal({1, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIs[cls][batchPos][proposal].y : 0.0;
                        mROIsBBOxFinal({2, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIs[cls][batchPos][proposal].w : 0.0;
                        mROIsBBOxFinal({3, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIs[cls][batchPos][proposal].h : 0.0;
                        mROIsBBOxFinal({4, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIs[cls][batchPos][proposal].s : 0.0;
                        mROIsMapAnchorsFinal({0, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIsAnchors[cls][batchPos][proposal].x : 0.0;
                        mROIsMapAnchorsFinal({1, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIsAnchors[cls][batchPos][proposal].y : 0.0;
                        mROIsMapAnchorsFinal({2, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIsAnchors[cls][batchPos][proposal].w : 0.0;
                        mROIsMapAnchorsFinal({3, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIsAnchors[cls][batchPos][proposal].h : 0.0;
                        mROIsMapAnchorsFinal({4, proposal, batchPos, cls}) = (proposal < nbElementNMS) ? ROIsAnchors[cls][batchPos][proposal].s : 0.0;

                    }
                    mROIsIndexFinal(batchPos, cls) = mNbProposals;

                }

            }
            else{
                mROIsIndexFinal(batchPos, cls) = 0;
                for(unsigned int proposal = 0; proposal < mNbProposals; ++ proposal )
                {
                    mROIsBBOxFinal({0, proposal, batchPos, cls}) = 0.0;
                    mROIsBBOxFinal({1, proposal, batchPos, cls}) = 0.0;
                    mROIsBBOxFinal({2, proposal, batchPos, cls}) = 0.0;
                    mROIsBBOxFinal({3, proposal, batchPos, cls}) = 0.0;
                    mROIsBBOxFinal({4, proposal, batchPos, cls}) = 0.0;
                    mROIsMapAnchorsFinal({0, proposal, batchPos, cls}) = 0.0;
                    mROIsMapAnchorsFinal({1, proposal, batchPos, cls}) = 0.0;
                    mROIsMapAnchorsFinal({2, proposal, batchPos, cls}) = 0.0;
                    mROIsMapAnchorsFinal({3, proposal, batchPos, cls}) = 0.0;
                    mROIsMapAnchorsFinal({4, proposal, batchPos, cls}) = 0.0;
                }
            }

        }
    }

    mROIsIndexFinal.synchronizeHToD();
    mROIsBBOxFinal.synchronizeHToD();
    mROIsMapAnchorsFinal.synchronizeHToD();

    for(unsigned int cls = 0; cls < mNbClass; ++cls)
    {
        const unsigned int yBlocks = (mNumParts.size() > 0) && (mNumTemplates.size() > 0) ? std::max(mNumParts[cls], mNumTemplates[cls]) : 1;
        dim3 outputs_blocks = {(unsigned int) std::ceil((float)mNbProposals/32.0), 
                                std::max(1U, yBlocks), 
                                (unsigned int) mInputs[0].dimB() } ;

        dim3 outputs_threads = {32, 1, 1};
        cudaS_SSD_output_gathering( mInputs.dimB(),
                                    mNbClass,
                                    mNbAnchors,
                                    mInputs[0].dimX(),
                                    mInputs[0].dimY(),
                                    mNbProposals,
                                    mROIsIndexFinal.getDevicePtr() + cls*mInputs.dimB(),
                                    cls,
                                    mNumParts.size() > 0 ? std::accumulate(mNumParts.begin(), mNumParts.end(), 0) : 0,
                                    mNumTemplates.size() > 0 ? std::accumulate(mNumTemplates.begin(), mNumTemplates.end(), 0) : 0,
                                    mNumParts.size() > 0 ? mMaxParts : 0,
                                    mNumTemplates.size() > 0 ? mMaxTemplates : 0,
                                    mNumParts.size() > 0 ? std::accumulate(mNumParts.begin(), mNumParts.begin() + cls, 0) * 2 * mNbAnchors : 0,
                                    mNumTemplates.size() > 0 ? std::accumulate(mNumTemplates.begin(), mNumTemplates.begin() + cls, 0) * 3 * mNbAnchors : 0,
                                    mNumParts.size() > 0 ? mNumParts[cls] : 0,
                                    mNumTemplates.size() > 0 ? mNumTemplates[cls] : 0,
                                    xRatio,
                                    yRatio,
                                    xOutputRatio,
                                    yOutputRatio,
                                    mROIsBBOxFinal.getDevicePtr() + cls*mInputs.dimB()*5*mNbProposals,
                                    mROIsMapAnchorsFinal.getDevicePtr() + cls*mInputs.dimB()*5*mNbProposals,
                                    mGPUAnchors.getDevicePtr(),
                                    (mNumParts.size() > 0 && mNumTemplates.size() > 0) ?  input_parts->getDevicePtr(): mGPUAnchors.getDevicePtr(),
                                    (mNumParts.size() > 0 && mNumTemplates.size() > 0) ?  input_templates->getDevicePtr() : mGPUAnchors.getDevicePtr(),
                                    mOutputs.getDevicePtr(),
                                    outputs_blocks,
                                    outputs_threads);
    }
    Cell_Frame_CUDA<Float_T>::propagate();
    mDiffInputs.clearValid();

}

void N2D2::ObjectDetCell_Frame_CUDA::backPropagate()
{
    // No backpropagation for this layer
}

void N2D2::ObjectDetCell_Frame_CUDA::update()
{
    // Nothing to update
}

void N2D2::ObjectDetCell_Frame_CUDA::setOutputsDims()
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
#endif