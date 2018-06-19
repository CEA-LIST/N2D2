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
        std::vector<std::pair<Tensor<int>::Index, Float_T> > ROIs;
        ROIs.reserve(mNbAnchors * mInputs[0].dimY() * mInputs[0].dimX());

        for (unsigned int k = 0; k < mNbAnchors; ++k) {
            for (unsigned int y = 0; y < mInputs[0].dimY(); ++y) {
                for (unsigned int x = 0; x < mInputs[0].dimX(); ++x) {
                    const Float_T value = mInputs(x,
                                                  y,
                                                  k + mScoreIndex * mNbAnchors,
                                                  batchPos);
                    const Float_T w = mInputs(x,
                                              y,
                                              k + 3 * mNbAnchors,
                                              batchPos);
                    const Float_T h = mInputs(x,
                                              y,
                                              k + 4 * mNbAnchors,
                                              batchPos);

                    if (value >= 0.0 && w >= mMinWidth && h >= mMinHeight) {
                        ROIs.push_back(std::make_pair(
                            Tensor<int>::Index(x, y, k, batchPos), value));
                    }
                }
            }
        }

        // Sort ROIs by value
        if (mPre_NMS_TopN > 0 && mPre_NMS_TopN < ROIs.size()) {
            std::partial_sort(ROIs.begin(),
                              ROIs.begin() + mPre_NMS_TopN,
                              ROIs.end(),
                              Utils::PairSecondPred<Tensor<int>::Index,
                                Float_T, std::greater<Float_T> >());

            // Drop the lowest score (unsorted) ROIs
            ROIs.resize(mPre_NMS_TopN);
        }
        else {
            std::sort(ROIs.begin(),
                      ROIs.end(),
                      Utils::PairSecondPred<Tensor<int>::Index, Float_T,
                        std::greater<Float_T> >());
        }

        if (inference) {
            // Non-Maximum Suppression (NMS)

            // This implementation is inspired by D. Oro, C. Fern, X. Martorell
            // and J. Hernando, "WORK-EFFICIENT PARALLEL NON-MAXIMUM SUPPRESSION
            // FOR EMBEDDED GPU ARCHITECTURES"
            // http://rapid-project.eu/_docs/icassp2016.pdf

            Tensor<int> invalid({ROIs.size(), ROIs.size()}, 0);

            // Flag
#pragma omp parallel for if (ROIs.size() > 16)
            for (int i = 0; i < (int)ROIs.size(); ++i) {
                const Tensor<int>::Index& ROIMax = ROIs[i].first;

                const Float_T x0 = mInputs(ROIMax[0],
                                           ROIMax[1],
                                           ROIMax[2] + 1 * mNbAnchors,
                                           ROIMax[3]);
                const Float_T y0 = mInputs(ROIMax[0],
                                           ROIMax[1],
                                           ROIMax[2] + 2 * mNbAnchors,
                                           ROIMax[3]);
                const Float_T w0 = mInputs(ROIMax[0],
                                           ROIMax[1],
                                           ROIMax[2] + 3 * mNbAnchors,
                                           ROIMax[3]);
                const Float_T h0 = mInputs(ROIMax[0],
                                           ROIMax[1],
                                           ROIMax[2] + 4 * mNbAnchors,
                                           ROIMax[3]);

                for (unsigned int j = i + 1; j < ROIs.size(); ++j) {
                    const Tensor<int>::Index& ROI = ROIs[j].first;

                    const Float_T x = mInputs(ROI[0],
                                              ROI[1],
                                              ROI[2] + 1 * mNbAnchors,
                                              ROI[3]);
                    const Float_T y = mInputs(ROI[0],
                                              ROI[1],
                                              ROI[2] + 2 * mNbAnchors,
                                              ROI[3]);
                    const Float_T w = mInputs(ROI[0],
                                              ROI[1],
                                              ROI[2] + 3 * mNbAnchors,
                                              ROI[3]);
                    const Float_T h = mInputs(ROI[0],
                                              ROI[1],
                                              ROI[2] + 4 * mNbAnchors,
                                              ROI[3]);

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
                            // Row j is invalid at column i
                            // = ROI j is invalid because it overlaps with ROI i
                            // Thread-safe because i is != for every thread
                            invalid(j, i) = 1;
                        }
                    }
                }
            }

            // Reduction
            for (unsigned int i = 0, n = 0; i < ROIs.size() && n < mNbProposals;
                ++i)
            {
                // Check if any preceding ROI invalidates this ROI
                // => the row i from column 0 to i (excluded) must be valid
#if __cplusplus >= 201703L
                const int isInvalid
                    = std::reduce(std::execution::par_unseq,
                                  invalid[i].begin(),
                                  invalid[i].begin() + i,
                                  0);
#else
                int isInvalid = 0;
                    //= std::accumulate(invalid[i].begin(),
                    //                  invalid[i].begin() + i,
                    //                  0);
                for(unsigned int k = 0; k < i; ++k)
                    if(invalid(i,k) == 1)
                        isInvalid += 1;
#endif

                if (isInvalid) {
                    // Main difference with D. Oro et al. reference:
                    // our implementation is NOT greedy

                    // ROI i is invalid:
                    // => ROI i cannot invalidate another ROI anymore
                    // => clear all following rows at column i
                    for (unsigned int j = i + 1; j < ROIs.size(); ++j)
                        invalid(j, i) = 0;
                }
                else {
                    mOutputs(0, n + batchPos * mNbProposals)
                        = mInputs(ROIs[i].first[0],
                                  ROIs[i].first[1],
                                  ROIs[i].first[2] + 1 * mNbAnchors,
                                  ROIs[i].first[3]);
                    mOutputs(1, n + batchPos * mNbProposals)
                        = mInputs(ROIs[i].first[0],
                                  ROIs[i].first[1],
                                  ROIs[i].first[2] + 2 * mNbAnchors,
                                  ROIs[i].first[3]);
                    mOutputs(2, n + batchPos * mNbProposals)
                        = mInputs(ROIs[i].first[0],
                                  ROIs[i].first[1],
                                  ROIs[i].first[2] + 3 * mNbAnchors,
                                  ROIs[i].first[3]);
                    mOutputs(3, n + batchPos * mNbProposals)
                        = mInputs(ROIs[i].first[0],
                                  ROIs[i].first[1],
                                  ROIs[i].first[2] + 4 * mNbAnchors,
                                  ROIs[i].first[3]);
                    mAnchors[n + batchPos * mNbProposals] = ROIs[i].first;

                    ++n;
                }
            }

/*
            // DEBUG
            std::cout << "RPCell NMS: " << ROIs.size() << " ROIs out of "
                << (mNbAnchors * mInputs[0].dimX() * mInputs[0].dimY())
                << " remaining" << std::endl;

            for (unsigned int i = 0,
                size = std::min(5, (int)ROIs.size() - 1);
                i < size; ++i)
            {
                std::cout << "  " << ROIs[i].first[0]
                    << "x" << ROIs[i].first[1] << ": " << ROIs[i].second
                    << std::endl;
            }
*/
        }
        else {
/*
            // DEBUG
            std::cout << "Top-5 IoU ROIs:" << std::endl;

            for (int i = ROIs.size() - 1; i >= 0
                && i >= (int)ROIs.size() - 5;
                --i)
            {
                std::cout << "  " << ROIs[i].first[0] << "x"
                    << ROIs[i].first[1]
                    << ": " << ROIs[i].second << std::endl;
            }
*/
            const unsigned int nbForegroundROIs = Utils::round(mForegroundRate
                                                            * mNbProposals);
            const unsigned int nbBackgroundROIs = mNbProposals
                - (int)nbForegroundROIs;

            std::vector<Tensor<int>::Index> foregroundIoU;
            std::vector<Tensor<int>::Index> backgroundIoU;
            std::vector<Tensor<int>::Index> remainingIoU;

            for (unsigned int i = 0, size = ROIs.size(); i < size; ++i) {
                const Tensor<int>::Index& ROI = ROIs[i].first;
                const Float_T IoU = mInputs(ROI[0],
                                            ROI[1],
                                            ROI[2] + mIoUIndex * mNbAnchors,
                                            ROI[3]);

                if (IoU >= mForegroundMinIoU)
                    foregroundIoU.push_back(ROI);
                else if (IoU > mBackgroundMinIoU && IoU < mBackgroundMaxIoU)
                    backgroundIoU.push_back(ROI);
                else
                    remainingIoU.push_back(ROI);
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
                std::vector<Tensor<int>::Index>& IoU =
                                        (n < nbForegroundROIs)
                                            ? foregroundIoU
                                            : backgroundIoU;

                const unsigned int idx = Random::randUniform(0, IoU.size() - 1);

                mOutputs(0, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx][0],
                              IoU[idx][1],
                              IoU[idx][2] + 1 * mNbAnchors,
                              batchPos);
                mOutputs(1, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx][0],
                              IoU[idx][1],
                              IoU[idx][2] + 2 * mNbAnchors,
                              batchPos);
                mOutputs(2, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx][0],
                              IoU[idx][1],
                              IoU[idx][2] + 3 * mNbAnchors,
                              batchPos);
                mOutputs(3, n + batchPos * mNbProposals)
                    = mInputs(IoU[idx][0],
                              IoU[idx][1],
                              IoU[idx][2] + 4 * mNbAnchors,
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

void N2D2::RPCell_Frame::setOutputsDims()
{
    RPCell::setOutputsDims();

    if (mOutputs.empty()) {
        mOutputs.resize({mOutputsDims[0],
                        mOutputsDims[1],
                        getNbOutputs(),
                        mNbProposals * mInputs.dimB()});
        mDiffInputs.resize({mOutputsDims[0],
                           mOutputsDims[1],
                           getNbOutputs(),
                           mNbProposals * mInputs.dimB()});
    }
}
