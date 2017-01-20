/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#include "ComputerVision/LSL_Box.hpp"

void N2D2::ComputerVision::LSL_Box::process(const Matrix<unsigned char>& frame,
                                            int cls)
{
    const unsigned int width = frame.cols();
    const unsigned int height = frame.rows();

    std::vector<unsigned int> lineLabels, prevLineLabels;
    std::vector<unsigned int> lineAbsLabels, prevLineAbsLabels;
    std::vector<unsigned int> eqLabels;

    lineLabels.reserve(width);
    prevLineLabels.resize(width, 0); // Must be initialized if segments are
    // found in the first line

    // mRoi MUST be cleared everytime because its size is used as absolute label
    mRoi.clear();

    for (unsigned int i = 0; i < height; ++i) {
        // Step #1: relative labeling of each line
        //////////////////////////////////////////

        std::vector<unsigned int> lineSegments; // max size = width + 1

        unsigned int nbLineRelLabel = 0; // initial label
        bool prevPixelVal = false; // previous value of pixel = background

        for (unsigned int j = 0; j < width; ++j) {
            const bool pixelVal = (bool)frame(i, j);

            if (pixelVal != prevPixelVal) {
                // Begin/end of segment
                // pixelVal = 1 => begin of segment, prevPixelVal = 0
                // pixelVal = 0 => end of segment, prevPixelVal = 1
                lineSegments.push_back(j - prevPixelVal); //  store
                ++nbLineRelLabel; // segment label incrementation
            }

            lineLabels.push_back(nbLineRelLabel);
            prevPixelVal = pixelVal;
        }

        if (prevPixelVal) {
            // Close the segment
            lineSegments.push_back(width - 1);
        }

        // Step #2: equivalence and ROIs construction
        /////////////////////////////////////////////

        for (unsigned int lineRelLabel = 1; lineRelLabel <= nbLineRelLabel;
             lineRelLabel += 2) {
            // Boundaries [j0,j1] of the segment lineRelLabel
            unsigned int j0 = lineSegments[lineRelLabel - 1];
            unsigned int j1 = lineSegments[lineRelLabel];

            const ROI::Roi_T lineRoi = ROI::Roi_T(i, j0, i, j1, cls);

            // check extension in case of 8-connect algorithm
            if (j0 > 0)
                --j0;
            if (j1 < width - 1)
                ++j1;

            // relative labels of every adjacent segment in the previous line:
            // prevLineLabel0 is the label of the first segment and
            // prevLineLabel1 the label of the last segment
            int prevLineLabel0 = prevLineLabels[j0];
            int prevLineLabel1 = prevLineLabels[j1];

            // background slices are labeled with even numbers
            // check label parity: segments are odd
            if (prevLineLabel0 % 2 == 0)
                ++prevLineLabel0;
            if (prevLineLabel1 % 2 == 0)
                --prevLineLabel1; // prevLineLabel1 can be equal to -1

            if (prevLineLabel1 >= prevLineLabel0) {
                unsigned int labelAbsMin
                    = prevLineAbsLabels[(prevLineLabel0 - 1) / 2];
                unsigned int eqLabelMin = eqLabels[labelAbsMin];

                for (unsigned int prevLineLabel = prevLineLabel0 + 2;
                     prevLineLabel <= (unsigned int)prevLineLabel1;
                     prevLineLabel += 2) {
                    const unsigned int labelAbs
                        = prevLineAbsLabels[(prevLineLabel - 1) / 2];
                    const unsigned int eqLabel = eqLabels[labelAbs];

                    // min extraction and propagation
                    if (eqLabelMin < eqLabel) {
                        eqLabels[labelAbs] = eqLabelMin;
                        mRoi[eqLabelMin]
                            = ROI::merge(mRoi[eqLabelMin], mRoi[labelAbs]);
                    } else {
                        eqLabelMin = eqLabel;
                        eqLabels[labelAbsMin] = eqLabelMin;
                        mRoi[eqLabelMin]
                            = ROI::merge(mRoi[eqLabelMin], mRoi[labelAbsMin]);
                        labelAbsMin = labelAbs;
                    }
                }

                mRoi[eqLabelMin] = ROI::merge(mRoi[eqLabelMin], lineRoi);
                lineAbsLabels.push_back(eqLabelMin);
            } else {
                const unsigned int labelAbs = mRoi.size();

                mRoi.push_back(lineRoi);
                eqLabels.push_back(labelAbs);
                lineAbsLabels.push_back(labelAbs);
            }
        }

        // Put lineLabels in prevLineLabels and clear lineLabels
        prevLineLabels.swap(lineLabels);
        lineLabels.clear();

        // Put lineAbsLabels in prevLineAbsLabels and clear lineAbsLabels
        prevLineAbsLabels.swap(lineAbsLabels);
        lineAbsLabels.clear();
    }

    // Remove ROIs that have an equivalence, meaning that they were merged to a
    // bigger ROI.
    // Start from the end of mRoi to avoid invalidating the positions of earlier
    // merged ROIs when a ROI is removed.
    for (int eqLabel = eqLabels.size() - 1; eqLabel >= 0; --eqLabel) {
        if (eqLabels[eqLabel] != (unsigned int)eqLabel)
            mRoi.erase(mRoi.begin() + eqLabel);
    }
}
