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

#ifndef N2D2_COMPUTERVISION_ROI_H
#define N2D2_COMPUTERVISION_ROI_H

#include <tuple>
#include <vector>

#include "containers/Matrix.hpp"

namespace N2D2 {
namespace ComputerVision {
    namespace ROI {
        struct Roi_T {
            Roi_T(unsigned int i0_,
                  unsigned int j0_,
                  unsigned int i1_,
                  unsigned int j1_,
                  int cls_ = 0)
                : i0(i0_), j0(j0_), i1(i1_), j1(j1_), cls(cls_) {};

            unsigned int i0;
            unsigned int j0;
            unsigned int i1;
            unsigned int j1;
            int cls;
        };

        /**
         * Merge two ROIs, resulting in a new ROI bigger size.
         *
         * @param a             First ROI to merge
         * @param b             Second ROI to merge
         * @return New ROI including a and b
        */
        inline Roi_T merge(const Roi_T& a, const Roi_T& b)
        {
            return Roi_T(std::min(a.i0, b.i0),
                         std::min(a.j0, b.j0),
                         std::max(a.i1, b.i1),
                         std::max(a.j1, b.j1),
                         a.cls);
        }

        /**
         * Remove ROIs of small size and aspect ratio (width/height) that is not
         *within a specific range.
         * If any of the four above conditions is not met, the ROI is removed.
         *
         * @param roi               Vector of ROI to process
         * @param minHeight         Minimum height of the ROI to keep it
         * @param minWidth          Minimum width of the ROI to keep it
         * @param minAspectRatio    Minimum aspect ratio (width/height) of the
         *ROI to keep it (default is 0 = no minimum)
         * @param maxAspectRatio    Maximum aspect ratio (width/height) of the
         *ROI to keep it (default is 0 = no maximum)
        */
        void filterSize(std::vector<Roi_T>& roi,
                        unsigned int minHeight,
                        unsigned int minWidth,
                        double minAspectRatio = 0.0,
                        double maxAspectRatio = 0.0);

        /**
         * Remove or merge ROIs that overlaps with other ones.
         *
         * @param roi               Vector of ROI to process
         * @param xMaxOverlap       Maximum relative horizontal overlapping
         *before remove/merge
         *                          Between 0 (= no overlap) and 1 (= full
         *overlap)
         * @param yMaxOverlap       Maximum relative vertical overlapping before
         *remove/merge (between 0 and 1)
         * @param merge             If true, the overlapping ROI is merged with
         *the overlapped one, resulting in a bigger ROI
        */
        void filterOverlapping(std::vector<Roi_T>& roi,
                               double xMaxOverlap = 0.5,
                               double yMaxOverlap = 0.5,
                               bool merge = false);

        /**
         * Merge ROIs that are close.
         *
         * @param roi               Vector of ROI to process
         * @param xMaxDist          Maximum horizontal distance for merging (in
         *pixels)
         * @param yMaxDist          Maximum vertical distance for merging (in
         *pixels)
        */
        void filterSeparability(std::vector<Roi_T>& roi,
                                unsigned int xMaxDist,
                                unsigned int yMaxDist);
    }
}
}

#endif // N2D2_COMPUTERVISION_ROI_H
