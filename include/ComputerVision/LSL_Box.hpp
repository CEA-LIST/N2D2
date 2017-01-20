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

#ifndef N2D2_COMPUTERVISION_LSL_BOX_H
#define N2D2_COMPUTERVISION_LSL_BOX_H

#include <set>
#include <tuple>
#include <vector>

#include "ComputerVision/ROI.hpp"
#include "containers/Matrix.hpp"

namespace N2D2 {
namespace ComputerVision {
    /**
     * Variation of the "Light Speed Labeling" algorithm that produces directly
     * the ROI boxes.
     * Only the current and previous lines are kept in memory at any time, thus
     * minimizing the required memory.
     * (NOT PUBLISHED)
    */
    class LSL_Box {
    public:
        LSL_Box(unsigned int minSize = 0) : mMinSize(minSize)
        {
        }
        template <class T> void process(const Matrix<T>& frame);
        void process(const Matrix<unsigned char>& frame, int cls = 0);
        const std::vector<ROI::Roi_T>& getRoi() const
        {
            return mRoi;
        };
        std::vector<ROI::Roi_T>& roi()
        {
            return mRoi;
        };

    private:
        // Extracted ROIs
        std::vector<ROI::Roi_T> mRoi;
        unsigned int mMinSize;
    };
}
}

template <class T>
void N2D2::ComputerVision::LSL_Box::process(const Matrix<T>& frame)
{
    // Find unique values
    std::set<T> uniqueValues;

    for (unsigned int index = 0; index < frame.size(); ++index) {
        if (frame(index) != 0)
            uniqueValues.insert(frame(index));
    }

    std::vector<ROI::Roi_T> roi;
    Matrix<unsigned char> frameCls(frame.rows(), frame.cols());

    for (typename std::set<T>::const_iterator it = uniqueValues.begin(),
                                              itEnd = uniqueValues.end();
         it != itEnd;
         ++it) {
        for (unsigned int index = 0; index < frame.size(); ++index)
            frameCls(index) = (frame(index) == (*it));

        process(frameCls, (*it));

        if (mMinSize > 0) {
            for (int k = mRoi.size() - 1; k >= 0; --k) {
                unsigned int size = 0;

                for (unsigned int i = mRoi[k].i0; i < mRoi[k].i1; ++i) {
                    for (unsigned int j = mRoi[k].j0; j < mRoi[k].j1; ++j)
                        size += (frameCls(i, j) != 0);
                }

                if (size < mMinSize)
                    mRoi.erase(mRoi.begin() + k);
            }
        }

        roi.insert(roi.end(), mRoi.begin(), mRoi.end());
    }

    mRoi.swap(roi);
}

#endif // N2D2_COMPUTERVISION_LSL_BOX_H
