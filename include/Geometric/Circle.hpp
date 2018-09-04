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

#ifndef N2D2_GEOMETRIC_CIRCLE_H
#define N2D2_GEOMETRIC_CIRCLE_H

#ifdef OPENCV_USE_OLD_HEADERS       //  before OpenCV 2.2.0
    #include "cv.h"
#else
    #include "opencv2/core/version.hpp"
    #if CV_MAJOR_VERSION == 2
        #include "opencv2/core/core.hpp"
        #include "opencv2/imgproc/imgproc.hpp"
    #elif CV_MAJOR_VERSION >= 3
        #include "opencv2/core.hpp"
        #include "opencv2/imgproc.hpp"
    #endif
#endif

namespace N2D2 {
namespace Geometric {
    template <class T> struct Circle {
        typedef cv::Point_<T> Point_T;
        Point_T center;
        double radius;

        Circle(const Point_T& center_, double radius_)
            : center(center_), radius(radius_) {};
        virtual ~Circle() {};
    };
}
}

#endif // N2D2_GEOMETRIC_CIRCLE_H
