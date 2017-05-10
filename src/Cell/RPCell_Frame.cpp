/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Cell/RPCell_Frame.hpp"

N2D2::Registrar<N2D2::RPCell>
N2D2::RPCell_Frame::mRegistrar("Frame", N2D2::RPCell_Frame::create);

N2D2::RPCell_Frame::RPCell_Frame(const std::string& name,
                                 unsigned int nbAnchors,
                                 unsigned int nbProposals,
                                 unsigned int scoreIndex,
                                 unsigned int IoUIndex)
    : Cell(name, 4),
      RPCell(name, nbAnchors, nbProposals, scoreIndex, IoUIndex),
      Cell_Frame(name, 4)
{
    // ctor
}

void N2D2::RPCell_Frame::initialize()
{
    mAnchors.resize(mNbProposals * mInputs.dimB());
}

void N2D2::RPCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDToH();

#pragma omp parallel for if (mInputs.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        // Collect all ROIs in the "ROIs" vector
        std::vector<std::pair<Tensor4d<int>::Index, Float_T> > ROIs;

        for (unsigned int k = 0; k < mNbAnchors; ++k) {
            for (unsigned int y = 0; y < mInputs[0].dimY(); ++y) {
                for (unsigned int x = 0; x < mInputs[0].dimX(); ++x) {
                    const Float_T value = mInputs(x,
                                                  y,
                                                  k + ((inference)
                                                       ? mScoreIndex
                                                       : mIoUIndex)
                                                    * mNbAnchors,
                                                  batchPos);

                    ROIs.push_back(std::make_pair(
                        Tensor4d<int>::Index(x, y, k, batchPos), value));
                }
            }
        }

        // Sort ROIs by value
        std::sort(ROIs.begin(),
                  ROIs.end(),
                  Utils::PairSecondPred<Tensor4d<int>::Index, Float_T>());

        if (inference) {
            // Non-Maximum Suppression (NMS)
            std::vector<std::pair<Tensor4d<int>::Index, Float_T> > NMS_ROIs;

            while (!ROIs.empty()) {
                const Tensor4d<int>::Index& ROIMax = ROIs.back().first;

                const Float_T x0 = mInputs(ROIMax.i,
                                           ROIMax.j,
                                           ROIMax.k + 1 * mNbAnchors,
                                           ROIMax.b);
                const Float_T y0 = mInputs(ROIMax.i,
                                           ROIMax.j,
                                           ROIMax.k + 2 * mNbAnchors,
                                           ROIMax.b);
                const Float_T w0 = mInputs(ROIMax.i,
                                           ROIMax.j,
                                           ROIMax.k + 3 * mNbAnchors,
                                           ROIMax.b);
                const Float_T h0 = mInputs(ROIMax.i,
                                           ROIMax.j,
                                           ROIMax.k + 4 * mNbAnchors,
                                           ROIMax.b);

                NMS_ROIs.push_back(ROIs.back());
                ROIs.pop_back();

                for (unsigned int i = 0; i < ROIs.size(); ) {
                    const Tensor4d<int>::Index& ROI = ROIs[i].first;

                    const Float_T x = mInputs(ROI.i,
                                              ROI.j,
                                              ROI.k + 1 * mNbAnchors,
                                              ROI.b);
                    const Float_T y = mInputs(ROI.i,
                                              ROI.j,
                                              ROI.k + 2 * mNbAnchors,
                                              ROI.b);
                    const Float_T w = mInputs(ROI.i,
                                              ROI.j,
                                              ROI.k + 3 * mNbAnchors,
                                              ROI.b);
                    const Float_T h = mInputs(ROI.i,
                                              ROI.j,
                                              ROI.k + 4 * mNbAnchors,
                                              ROI.b);

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
                            ROIs.erase(ROIs.begin() + i);
                            continue;
                        }
                    }

                    ++i;
                }
            }
/*
            // DEBUG
            std::cout << "RPCell NMS: " << NMS_ROIs.size() << " ROIs out of "
                << (mNbAnchors * mInputs[0].dimX() * mInputs[0].dimY())
                << " remaining" << std::endl;

            for (unsigned int i = 0,
                size = std::min(5, (int)NMS_ROIs.size() - 1);
                i < size; ++i)
            {
                std::cout << "  " << NMS_ROIs[i].first.i
                    << "x" << NMS_ROIs[i].first.j << ": " << NMS_ROIs[i].second
                    << std::endl;
            }
*/
            // Keep the top-N ROIs
            for (unsigned int n = 0; n < mNbProposals; ++n) {
                mOutputs(0, n + batchPos * mNbProposals)
                    = mInputs(NMS_ROIs[n].first.i,
                              NMS_ROIs[n].first.j,
                              NMS_ROIs[n].first.k + 1 * mNbAnchors,
                              NMS_ROIs[n].first.b);
                mOutputs(1, n + batchPos * mNbProposals)
                    = mInputs(NMS_ROIs[n].first.i,
                              NMS_ROIs[n].first.j,
                              NMS_ROIs[n].first.k + 2 * mNbAnchors,
                              NMS_ROIs[n].first.b);
                mOutputs(2, n + batchPos * mNbProposals)
                    = mInputs(NMS_ROIs[n].first.i,
                              NMS_ROIs[n].first.j,
                              NMS_ROIs[n].first.k + 3 * mNbAnchors,
                              NMS_ROIs[n].first.b);
                mOutputs(3, n + batchPos * mNbProposals)
                    = mInputs(NMS_ROIs[n].first.i,
                              NMS_ROIs[n].first.j,
                              NMS_ROIs[n].first.k + 4 * mNbAnchors,
                              NMS_ROIs[n].first.b);
                mAnchors[n + batchPos * mNbProposals] = NMS_ROIs[n].first;
            }
        }
        else {
/*
            // DEBUG
            std::cout << "Top-5 IoU ROIs:" << std::endl;

            for (int i = ROIs.size() - 1; i >= 0 && i >= (int)ROIs.size() - 5;
                --i)
            {
                std::cout << "  " << ROIs[i].first.i << "x" << ROIs[i].first.j
                    << ": " << ROIs[i].second << std::endl;
            }
*/
            const unsigned int nbForegroundROIs = Utils::round(mForegroundRate
                                                            * mNbProposals);
            const unsigned int nbBackgroundROIs = mNbProposals
                - (int)nbForegroundROIs;

            std::vector<Tensor4d<int>::Index> foregroundIoU;
            std::vector<Tensor4d<int>::Index> backgroundIoU;
            std::vector<Tensor4d<int>::Index> remainingIoU;

            for (unsigned int i = 0, size = ROIs.size(); i < size; ++i) {
                const Tensor4d<int>::Index& ROI = ROIs[i].first;
                const Float_T IoU = ROIs[i].second;

                if (IoU >= mForegroundMinIoU)
                    foregroundIoU.push_back(ROI);
                else if (IoU > mBackgroundMinIoU && IoU < mBackgroundMaxIoU)
                    backgroundIoU.push_back(ROI);
                else {
                    if (mInputs(ROI.i,
                              ROI.j,
                              ROI.k + 1 * mNbAnchors,
                              batchPos) >= 0
                        && mInputs(ROI.i,
                              ROI.j,
                              ROI.k + 2 * mNbAnchors,
                              batchPos) >= 0)
                    {
                        remainingIoU.push_back(ROI);
                    }
                }
            }
/*
            // DEBUG
            std::cout <<
                "Foreground ROIs: " << foregroundIoU.size() << "\n"
                "Background ROIs: " << backgroundIoU.size() << "\n"
                "Remaining ROIs: " << remainingIoU.size() << std::endl;
*/
            // Handle cases where there is not enough positive samples
            for (unsigned int i = 0,
                 size = std::max(0, (int)nbForegroundROIs
                                 - (int)foregroundIoU.size()),
                 remainingSize = remainingIoU.size() - 1;
                 i < size; ++i)
            {
                // Complete foregroundIoU with the highest IoU ROIs
                foregroundIoU.push_back(remainingIoU[remainingSize - i]);
            }

            // Handle cases where there is not enough negative samples
            for (unsigned int i = 0,
                 size = std::max(0, (int)nbBackgroundROIs
                                 - (int)backgroundIoU.size());
                 i < size; ++i)
            {
                // Complete backgroundIoU with the lowest IoU ROIs
                backgroundIoU.push_back(remainingIoU[i]);
            }
/*
            // DEBUG
            std::cout <<
                "Foreground ROIs (2): " << foregroundIoU.size() << "\n"
                "Background ROIs (2): " << backgroundIoU.size() << std::endl;
*/
            for (unsigned int n = 0; n < mNbProposals; ++n) {
                std::vector<Tensor4d<int>::Index>& IoU =
                                        (n < nbForegroundROIs)
                                            ? foregroundIoU
                                            : backgroundIoU;

                const unsigned int idx = Random::randUniform(0, IoU.size() - 1);

                mOutputs(0, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx].i,
                              IoU[idx].j,
                              IoU[idx].k + 1 * mNbAnchors,
                              batchPos);
                mOutputs(1, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx].i,
                              IoU[idx].j,
                              IoU[idx].k + 2 * mNbAnchors,
                              batchPos);
                mOutputs(2, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx].i,
                              IoU[idx].j,
                              IoU[idx].k + 3 * mNbAnchors,
                              batchPos);
                mOutputs(3, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx].i,
                              IoU[idx].j,
                              IoU[idx].k + 4 * mNbAnchors,
                              batchPos);
                mAnchors[n + batchPos * mNbProposals] = IoU[idx];

                IoU.erase(IoU.begin() + idx);
            }
        }
    }

    mDiffInputs.clearValid();
}

void N2D2::RPCell_Frame::backPropagate()
{
    // No backpropagation for this layer
}

void N2D2::RPCell_Frame::update()
{
    // Nothing to update
}

void N2D2::RPCell_Frame::setOutputsSize()
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
