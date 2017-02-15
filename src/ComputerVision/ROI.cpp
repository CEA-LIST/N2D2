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

#include "ComputerVision/ROI.hpp"

void N2D2::ComputerVision::ROI::filterSize(std::vector<Roi_T>& roi,
                                           unsigned int minHeight,
                                           unsigned int minWidth,
                                           double minAspectRatio,
                                           double maxAspectRatio)
{
    for (int i = roi.size() - 1; i >= 0; --i) {
        const unsigned int height = roi[i].i1 - roi[i].i0 + 1;
        const unsigned int width = roi[i].j1 - roi[i].j0 + 1;
        const double aspectRatio = width / (double)height;

        if (height < minHeight || width < minWidth
            || (minAspectRatio > 0.0 && aspectRatio < minAspectRatio)
            || (maxAspectRatio > 0.0 && aspectRatio > maxAspectRatio)) {
            roi.erase(roi.begin() + i);
        }
    }
}

void N2D2::ComputerVision::ROI::filterOverlapping(std::vector<Roi_T>& roi,
                                                  double xMaxOverlap,
                                                  double yMaxOverlap,
                                                  bool merge)
{
    for (int a = roi.size() - 1; a >= 0; --a) {
        const unsigned int aHeight = roi[a].i1 - roi[a].i0 + 1;
        const unsigned int aWidth = roi[a].j1 - roi[a].j0 + 1;
        const unsigned int aArea = aHeight * aWidth;

        for (int b = a - 1; b >= 0; --b) {
            if (roi[b].cls != roi[a].cls)
                continue;

            const unsigned int bHeight = roi[b].i1 - roi[b].i0 + 1;
            const unsigned int bWidth = roi[b].j1 - roi[b].j0 + 1;
            const unsigned int bArea = bHeight * bWidth;

            const int xOverlap
                = std::max(0,
                           (int)std::min(roi[a].j1, roi[b].j1)
                           - (int)std::max(roi[a].j0, roi[b].j0) + 1);
            const int yOverlap
                = std::max(0,
                           (int)std::min(roi[a].i1, roi[b].i1)
                           - (int)std::max(roi[a].i0, roi[b].i0) + 1);

            if (xOverlap * yOverlap > 0) {
                if (aArea <= bArea && xOverlap > xMaxOverlap * aWidth
                    && yOverlap > yMaxOverlap * aHeight) {
                    if (merge)
                        roi[b] = ROI::merge(roi[a], roi[b]);

                    roi.erase(roi.begin() + a);
                    break;
                } else if (aArea > bArea && xOverlap > xMaxOverlap * bWidth
                           && yOverlap > yMaxOverlap * bHeight) {
                    if (merge)
                        roi[a] = ROI::merge(roi[a], roi[b]);

                    roi.erase(roi.begin() + b);
                    --a;
                }
            }
        }
    }
}

void N2D2::ComputerVision::ROI::filterSeparability(std::vector<Roi_T>& roi,
                                                   unsigned int xMaxDist,
                                                   unsigned int yMaxDist)
{
    unsigned int nbROIs = 0;

    do {
        nbROIs = roi.size();

        for (int a = roi.size() - 1; a >= 0; --a) {
            for (int b = a - 1; b >= 0; --b) {
                if (roi[b].cls != roi[a].cls)
                    continue;

                const int xDist = (int)std::max(roi[a].j0, roi[b].j0)
                                  - (int)std::min(roi[a].j1, roi[b].j1);
                const int yDist = (int)std::max(roi[a].i0, roi[b].i0)
                                  - (int)std::min(roi[a].i1, roi[b].i1);

                if ((yDist <= 0 && xDist < (int)xMaxDist)
                    || (xDist <= 0 && yDist < (int)yMaxDist)) {
                    roi[b] = ROI::merge(roi[a], roi[b]);
                    roi.erase(roi.begin() + a);
                    break;
                }
            }
        }
    }
    while (roi.size() != nbROIs);
}
