/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_GEOMETRIC_POLYGON_H
#define N2D2_GEOMETRIC_POLYGON_H

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace N2D2 {
namespace Geometric {
    template <class T> struct Polygon {
        typedef cv::Point_<T> Point_T;
        std::vector<Point_T> points;

        Polygon(const std::vector<Point_T>& points_ = std::vector<Point_T>())
            : points(points_) {};
        virtual ~Polygon() {};
    };
}
}

#endif // N2D2_GEOMETRIC_POLYGON_H
