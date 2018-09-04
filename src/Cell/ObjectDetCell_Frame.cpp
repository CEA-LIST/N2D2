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

N2D2::Registrar<N2D2::ObjectDetCell>
N2D2::ObjectDetCell_Frame::mRegistrar("Frame", N2D2::ObjectDetCell_Frame::create);

N2D2::ObjectDetCell_Frame::ObjectDetCell_Frame(const std::string& name,
                                                StimuliProvider& sp,
                                                const unsigned int nbOutputs,
                                                unsigned int nbAnchors,
                                                unsigned int nbProposals,
                                                unsigned int nbClass,
                                                Float_T nmsThreshold,
                                                Float_T scoreThreshold)
    : Cell(name, nbOutputs),
      ObjectDetCell(name, sp, nbOutputs, nbAnchors, nbProposals, nbClass, nmsThreshold, scoreThreshold),
      Cell_Frame<Float_T>(name, nbOutputs)
{
    // ctor
}

void N2D2::ObjectDetCell_Frame::initialize()
{

}

void N2D2::ObjectDetCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDToH();
    const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[0]);

    const unsigned int inputBatch = mOutputs.dimB()/(mNbProposals*mNbClass);

    std::vector< std::vector< std::vector<BBox_T> > > ROIs;
    ROIs.resize(inputBatch);
    for(unsigned int batchPos = 0; batchPos < inputBatch; ++batchPos)
    {

        ROIs[batchPos].resize(mNbClass);
        unsigned int nbRoiDetected = 0;

        for(unsigned int cls = 0; cls < mNbClass; ++ cls)
        {
            for (unsigned int anchor = 0; anchor < mNbAnchors; ++anchor)
            {
                for (unsigned int y = 0; y < input.dimY(); ++y) {
                    for (unsigned int x = 0; x < input.dimX(); ++x) {

                        const Float_T value = input(x,
                                                     y,
                                                     anchor + cls*mNbAnchors,
                                                     batchPos);

                        if(value >= mScoreThreshold)
                        {
                            const unsigned int offset = mNbAnchors*mNbClass;

                            const Float_T xbbEst = input(x, y, anchor + cls*mNbAnchors + offset, batchPos);
                            const Float_T ybbEst = input(x, y, anchor + cls*mNbAnchors + 2*offset, batchPos);
                            const Float_T wbbEst = input(x, y, anchor + cls*mNbAnchors + 3*offset, batchPos);
                            const Float_T hbbEst = input(x, y, anchor + cls*mNbAnchors + 4*offset, batchPos);

                            ROIs[batchPos][cls].push_back(BBox_T(xbbEst,ybbEst,wbbEst,hbbEst));
                        }

                    }
                }

            }

            if(ROIs[batchPos][cls].size() > 0)
            {
                // Non-Maximum Suppression (NMS)
                for (unsigned int i = 0; i < ROIs[batchPos][cls].size() - 1; ++i)
                {
                    const Float_T x0 = ROIs[batchPos][cls][i].x;
                    const Float_T y0 = ROIs[batchPos][cls][i].y;
                    const Float_T w0 = ROIs[batchPos][cls][i].w;
                    const Float_T h0 = ROIs[batchPos][cls][i].h;

                    for (unsigned int j = i + 1; j < ROIs[batchPos][cls].size(); ) {

                        const Float_T x = ROIs[batchPos][cls][j].x;
                        const Float_T y = ROIs[batchPos][cls][j].y;
                        const Float_T w = ROIs[batchPos][cls][j].w;
                        const Float_T h = ROIs[batchPos][cls][j].h;

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
                                ROIs[batchPos][cls].erase(ROIs[batchPos][cls].begin() + j);
                                continue;
                            }
                        }
                        ++j;
                    }
                }
            }

            nbRoiDetected += ROIs[batchPos][cls].size();
        }


        for (unsigned int cls = 0; cls < mNbClass; ++cls)
        {
            unsigned int totalIdxPerClass = 0;

            for(unsigned int i = 0; i < ROIs[batchPos][cls].size() && totalIdxPerClass < mNbProposals; ++i)
            {
                const unsigned int n = totalIdxPerClass + cls*mNbProposals + batchPos*mNbProposals*mNbClass;
                mOutputs(0, n) = ROIs[batchPos][cls][i].x;
                mOutputs(1, n) = ROIs[batchPos][cls][i].y;
                mOutputs(2, n) = ROIs[batchPos][cls][i].w;
                mOutputs(3, n) = ROIs[batchPos][cls][i].h;
                mOutputs(4, n) = (float) cls;

                totalIdxPerClass++;
            }

            for(unsigned int rest = totalIdxPerClass; rest < mNbProposals; ++rest)
            {
                    const unsigned int n = rest + cls*mNbProposals +  batchPos*mNbProposals*mNbClass;
                    mOutputs(0, n) = 0.0;
                    mOutputs(1, n) = 0.0;
                    mOutputs(2, n) = 0.0;
                    mOutputs(3, n) = 0.0;
                    mOutputs(4, n) = 0.0;

            }

        }

    }


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
