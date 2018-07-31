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
#include <thrust/device_ptr.h>

N2D2::Registrar<N2D2::ObjectDetCell>
N2D2::ObjectDetCell_Frame_CUDA::mRegistrar("Frame_CUDA", N2D2::ObjectDetCell_Frame_CUDA::create);



N2D2::ObjectDetCell_Frame_CUDA::ObjectDetCell_Frame_CUDA(const std::string& name,
                                                StimuliProvider& sp,
                                                const unsigned int nbOutputs,
                                                unsigned int nbAnchors,
                                                unsigned int nbProposals,
                                                unsigned int nbClass,
                                                Float_T nmsThreshold,
                                                Float_T scoreThreshold)
    : Cell(name, nbOutputs),
      ObjectDetCell(name, sp, nbOutputs, nbAnchors, nbProposals, nbClass, nmsThreshold, scoreThreshold),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
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
}

void N2D2::ObjectDetCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();
    std::shared_ptr<CudaDeviceTensor<Float_T> > input
            = cuda_device_tensor_cast_nocopy<Float_T>(mInputs[0]);

    mOutputs.synchronizeDToH();
    const unsigned int inputBatchOffset = mInputs[0].dimX()*mInputs[0].dimY()*mInputs[0].dimZ();

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
                       mScoreThreshold,
                       input->getDevicePtr(),
                       mPixelMap.getDevicePtr(),
                       GPU_BLOCK_GRID[0],
                       GPU_THREAD_GRID[0]);

    for(unsigned int batchPos = 0; batchPos < mInputs.dimB(); ++batchPos)
    {
        unsigned int pixelOffset = mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors*mNbClass*batchPos;

        for(unsigned int cls = 0; cls < mNbClass; ++cls)
        {
            copy_if( mPixelMap.getDevicePtr() + pixelOffset,
                     mPixelMapSorted.getDevicePtr() + pixelOffset, 
                     mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors);

            pixelOffset += mInputs[0].dimX()*mInputs[0].dimY()*mNbAnchors;
        }
    }

    mPixelMapSorted.synchronizeDToH();
    std::vector<std::vector <unsigned int> > count(mInputs[0].dimB(), 
                                                   std::vector<unsigned int>(mNbClass));
    unsigned int totalValidPixel = 0;
    for(unsigned int cls = 0; cls < mNbClass; ++cls)
    {
        for(unsigned int batchPos = 0; batchPos < mInputs.dimB(); ++batchPos)
        {
            for(unsigned int anchor = 0; anchor < mNbAnchors; ++anchor)
            {
                for(unsigned int y = 0; y < mInputs[0].dimY(); ++y)
                {
                    for(unsigned int x = 0; x < mInputs[0].dimX(); ++x)
                    {
                        if(mPixelMapSorted(x, y, anchor, cls, batchPos) > -1)
                        {
                            ++count[batchPos][cls];
                        }
                        else
                            goto restartLoop;

                    }
                }
            }        
            restartLoop: 
            totalValidPixel += count[batchPos][cls];
        }
    }

    for(unsigned int cls = 0; cls < mNbClass; ++cls)
    {
        const int offset = cls*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY();

        for(unsigned int batchPos = 0; batchPos < mInputs.dimB(); ++batchPos)
        {
            const int batchOffset = batchPos*inputBatchOffset;
          
            unsigned int totalIdxPerClass = 0;
            if(count[batchPos][cls] > 0)
            {
                const int offsetBase = mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY();

                mPixelMapSorted.synchronizeHToD(0,
                                                0, 
                                                0, 
                                                cls,
                                                batchPos, mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY());

                mX_index.synchronizeDToH();
                mX_index.resize({count[batchPos][cls]}, -1);
                mX_index.assign({count[batchPos][cls]}, -1);

                mY_index.synchronizeDToH();
                mY_index.resize({count[batchPos][cls]}, -1);
                mY_index.assign({count[batchPos][cls]}, -1);

                mW_index.synchronizeDToH();
                mW_index.resize({count[batchPos][cls]}, -1);
                mW_index.assign({count[batchPos][cls]}, -1);

                mH_index.synchronizeDToH();
                mH_index.resize({count[batchPos][cls]}, -1);
                mH_index.assign({count[batchPos][cls]}, -1);

                mX_index.synchronizeHToD();
                mY_index.synchronizeHToD();
                mW_index.synchronizeHToD();
                mH_index.synchronizeHToD();

                thrust_gather(mPixelMapSorted.getDevicePtr() + offset + batchPos*mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY(), 
                            input->getDevicePtr() + offsetBase + offset + batchOffset,
                            mX_index.getDevicePtr(), 
                            count[batchPos][cls],
                            0,
                            0);
                thrust_gather(mPixelMapSorted.getDevicePtr() + offset + batchPos*mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY(), 
                            input->getDevicePtr() + 2*offsetBase + offset + batchOffset,
                            mY_index.getDevicePtr(), 
                            count[batchPos][cls],
                            0,
                            0);

                thrust_gather(mPixelMapSorted.getDevicePtr() + offset + batchPos*mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY(), 
                            input->getDevicePtr() + 3*offsetBase + offset + batchOffset,
                            mW_index.getDevicePtr(), 
                            count[batchPos][cls],
                            0,
                            0);

                thrust_gather(mPixelMapSorted.getDevicePtr() + offset + batchPos*mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY(), 
                            input->getDevicePtr() + 4*offsetBase + offset + batchOffset,
                            mH_index.getDevicePtr(), 
                            count[batchPos][cls],
                            0,
                            0);

                mX_index.synchronizeDToH();
                mY_index.synchronizeDToH();
                mW_index.synchronizeDToH();
                mH_index.synchronizeDToH();
                

                std::vector<BBox_T> ROIs;
                for(unsigned int idx = 0; idx < count[batchPos][cls]; ++idx)
                {
                    ROIs.push_back(BBox_T(  mX_index(idx),
                                            mY_index(idx),
                                            mW_index(idx),
                                            mH_index(idx)));
                }

                // Non-Maximum Suppression (NMS)
                for (unsigned int i = 0; i < ROIs.size() - 1; ++i)
                {
                    const Float_T x0 = ROIs[i].x;
                    const Float_T y0 = ROIs[i].y;
                    const Float_T w0 = ROIs[i].w;
                    const Float_T h0 = ROIs[i].h;

                    for (unsigned int j = i + 1; j < ROIs.size(); ) {

                        const Float_T x = ROIs[j].x;
                        const Float_T y = ROIs[j].y;
                        const Float_T w = ROIs[j].w;
                        const Float_T h = ROIs[j].h;

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
                                ROIs.erase(ROIs.begin() + j);
                                continue;
                            }
                        }
                        ++j;
                    }
                }

                for(unsigned int i = 0; i < ROIs.size() && i < mNbProposals; ++i)
                {
                    const unsigned int n = i + cls*mNbProposals + batchPos*mNbProposals*mNbClass;
                    mOutputs(0, n) = ROIs[i].x;
                    mOutputs(1, n) = ROIs[i].y;
                    mOutputs(2, n) = ROIs[i].w;
                    mOutputs(3, n) = ROIs[i].h;
                    mOutputs(4, n) = (float) cls;  
                    ++totalIdxPerClass;       

                }
            }     
            for(unsigned int rest = totalIdxPerClass; rest < mNbProposals; ++rest)
            {
                    const unsigned int n = rest + cls*mNbProposals + batchPos*mNbProposals*mNbClass;
                    mOutputs(0, n) = 0.0;
                    mOutputs(1, n) = 0.0;
                    mOutputs(2, n) = 0.0;
                    mOutputs(3, n) = 0.0;
                    mOutputs(4, n) = 0.0;

            } 

        }
    }

    mOutputs.synchronizeHToD();

    mDiffInputs.clearValid();
/*
    for(unsigned int cls = 0; cls < mNbClass; ++cls)
    {
        const int offset = cls*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY();

        for(unsigned int batchPos = 0; batchPos < mInputs.dimB(); ++batchPos)
        {
            for(unsigned int i = 0; i < mPixelMapSorted.size(); ++i)
            {
                if(mPixelMapSorted(i) > -1) 
                    ++count;
                else
                    break;
            }
            unsigned int totalIdxPerClass = 0;
            if(count > 0)
            {
                const int offsetBase = mNbClass*mNbAnchors*mInputs[0].dimX()*mInputs[0].dimY();

                mPixelMapSorted.resize({count});
                mPixelMapSorted.synchronizeHToD();

                mX_index.synchronizeDToH();
                mX_index.resize({count}, -1);
                mX_index.assign({count}, -1);

                mY_index.synchronizeDToH();
                mY_index.resize({count}, -1);
                mY_index.assign({count}, -1);

                mW_index.synchronizeDToH();
                mW_index.resize({count}, -1);
                mW_index.assign({count}, -1);

                mH_index.synchronizeDToH();
                mH_index.resize({count}, -1);
                mH_index.assign({count}, -1);

                mX_index.synchronizeHToD();
                mY_index.synchronizeHToD();
                mW_index.synchronizeHToD();
                mH_index.synchronizeHToD();

                thrust_gather(mPixelMapSorted.getDevicePtr(), 
                            mInputs[0].getDevicePtr() + offsetBase + offset,
                            mX_index.getDevicePtr(), 
                            count,
                            0,
                            0);
                thrust_gather(mPixelMapSorted.getDevicePtr(), 
                            mInputs[0].getDevicePtr() + 2*offsetBase + offset,
                            mY_index.getDevicePtr(), 
                            count,
                            0,
                            0);

                thrust_gather(mPixelMapSorted.getDevicePtr(), 
                            mInputs[0].getDevicePtr() + 3*offsetBase + offset,
                            mW_index.getDevicePtr(), 
                            count,
                            0,
                            0);

                thrust_gather(mPixelMapSorted.getDevicePtr(), 
                            mInputs[0].getDevicePtr() + 4*offsetBase + offset,
                            mH_index.getDevicePtr(), 
                            count,
                            0,
                            0);

                mX_index.synchronizeDToH();
                mY_index.synchronizeDToH();
                mW_index.synchronizeDToH();
                mH_index.synchronizeDToH();
                

                std::vector<BBox_T> ROIs;
                for(unsigned int idx = 0; idx < count; ++idx)
                {
                    ROIs.push_back(BBox_T(  mX_index(idx),
                                            mY_index(idx),
                                            mW_index(idx),
                                            mH_index(idx)));
                }

                // Non-Maximum Suppression (NMS)
                for (unsigned int i = 0; i < ROIs.size() - 1; ++i)
                {
                    const Float_T x0 = ROIs[i].x;
                    const Float_T y0 = ROIs[i].y;
                    const Float_T w0 = ROIs[i].w;
                    const Float_T h0 = ROIs[i].h;

                    for (unsigned int j = i + 1; j < ROIs.size(); ) {

                        const Float_T x = ROIs[j].x;
                        const Float_T y = ROIs[j].y;
                        const Float_T w = ROIs[j].w;
                        const Float_T h = ROIs[j].h;

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
                                ROIs.erase(ROIs.begin() + j);
                                continue;
                            }
                        }
                        ++j;
                    }
                }

                for(unsigned int i = 0; i < ROIs.size() && i < mNbProposals; ++i)
                {
                    const unsigned int n = i + cls*mNbProposals;
                    mOutputs(0, n) = ROIs[i].x;
                    mOutputs(1, n) = ROIs[i].y;
                    mOutputs(2, n) = ROIs[i].w;
                    mOutputs(3, n) = ROIs[i].h;
                    mOutputs(4, n) = (float) cls;  
                    ++totalIdxPerClass;       

                }
            }     
            for(unsigned int rest = totalIdxPerClass; rest < mNbProposals; ++rest)
            {
                    const unsigned int n = rest + cls*mNbProposals;
                    mOutputs(0, n) = 0.0;
                    mOutputs(1, n) = 0.0;
                    mOutputs(2, n) = 0.0;
                    mOutputs(3, n) = 0.0;
                    mOutputs(4, n) = 0.0;

            }   
        }
        mOutputs.synchronizeHToD();

    }


    mDiffInputs.clearValid();
*/
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