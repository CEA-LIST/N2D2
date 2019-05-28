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

#ifndef N2D2_GEOMETRIC_RECTANGLE_H
#define N2D2_GEOMETRIC_RECTANGLE_H

#include "Geometric/Polygon.hpp"

namespace N2D2 {
namespace Geometric {
    template <class T> struct Rectangle : public virtual Polygon<T> {
        // br must be exclusive, to be consistent with OpenCV Rect()
        Rectangle(const typename Geometric::Polygon<T>::Point_T& pt1,
                  const typename Geometric::Polygon<T>::Point_T& pt2);
        Rectangle(const typename Geometric::Polygon<T>::Point_T& tl,
                  unsigned int width,
                  unsigned int height);
        virtual ~Rectangle() {};

        using Geometric::Polygon<T>::points;
    };
}
}

template <class T>
N2D2::Geometric::Rectangle
    <T>::Rectangle(const typename Geometric::Polygon<T>::Point_T& pt1,
                   const typename Geometric::Polygon<T>::Point_T& pt2)
    : Geometric::Polygon<T>()
{
    int tl_x = std::min(pt1.x, pt2.x);
    int tl_y = std::min(pt1.y, pt2.y);
    int br_x = std::max(pt1.x, pt2.x);
    int br_y = std::max(pt1.y, pt2.y);

    // br is assumed to be exclusive, to be consistent with OpenCV Rect().
    // Internally, br must be inclusive (to work with e.g. cv::fillPoly())
    if (br_x > tl_x && br_y > tl_y) {
        points.resize(4);
        points[0] = typename Geometric::Polygon
            <T>::Point_T(tl_x, tl_y); // Top left
        points[1] = typename Geometric::Polygon
            <T>::Point_T(br_x - 1, tl_y); // Top right (inclusive)
        points[2] = typename Geometric::Polygon
            <T>::Point_T(br_x - 1, br_y - 1); // Bottom right (inclusive)
        points[3] = typename Geometric::Polygon
            <T>::Point_T(tl_x, br_y - 1); // Bottom left (inclusive)
    }
}

template <class T>
N2D2::Geometric::Rectangle
    <T>::Rectangle(const typename Geometric::Polygon<T>::Point_T& tl,
                   unsigned int width,
                   unsigned int height)
    : Geometric::Polygon<T>()
{
    if (width > 0 && height > 0) {
        points.resize(4);
        points[0] = tl; // Top left
        points[1] = typename Geometric::Polygon
            <T>::Point_T(tl.x + width - 1, tl.y); // Top right (inclusive)
        points[2] = typename Geometric::Polygon
            <T>::Point_T(tl.x + width - 1, tl.y + height - 1);
                // Bottom right (inclusive)
        points[3] = typename Geometric::Polygon
            <T>::Point_T(tl.x, tl.y + height - 1); // Bottom left (inclusive)
    }
}

#endif // N2D2_GEOMETRIC_RECTANGLE_H
