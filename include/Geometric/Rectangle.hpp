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
        Rectangle(const typename Geometric::Polygon<T>::Point_T& tl,
                  const typename Geometric::Polygon<T>::Point_T& br);
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
    <T>::Rectangle(const typename Geometric::Polygon<T>::Point_T& tl,
                   const typename Geometric::Polygon<T>::Point_T& br)
    : Geometric::Polygon<T>()
{
    points.resize(4);
    points[0] = tl; // Top left
    points[1] = typename Geometric::Polygon
        <T>::Point_T(br.x, tl.y); // Top right
    points[2] = br; // Bottom right
    points[3] = typename Geometric::Polygon
        <T>::Point_T(tl.x, br.y); // Bottom left
}

template <class T>
N2D2::Geometric::Rectangle
    <T>::Rectangle(const typename Geometric::Polygon<T>::Point_T& tl,
                   unsigned int width,
                   unsigned int height)
    : Geometric::Polygon<T>()
{
    points.resize(4);
    points[0] = tl; // Top left
    points[1] = typename Geometric::Polygon
        <T>::Point_T(tl.x + width, tl.y); // Top right
    points[2] = typename Geometric::Polygon
        <T>::Point_T(tl.x + width, tl.y + height); // Bottom right
    points[3] = typename Geometric::Polygon
        <T>::Point_T(tl.x, tl.y + height); // Bottom left
}

#endif // N2D2_GEOMETRIC_RECTANGLE_H
