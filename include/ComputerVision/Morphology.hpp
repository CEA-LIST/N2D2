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

#ifndef N2D2_COMPUTERVISION_MORPHOLOGY_H
#define N2D2_COMPUTERVISION_MORPHOLOGY_H

#include <tuple>
#include <vector>

#include "containers/Matrix.hpp"

namespace N2D2 {
namespace ComputerVision {
    enum Morphology {
        Erode,
        Dilate
    };
    template <class T>
    Matrix<T> morphology(const Matrix<T>& frame,
                         const Matrix<unsigned char>& filter,
                         Morphology type);
}
}

template <class T>
N2D2::Matrix<T> N2D2::ComputerVision::morphology(const Matrix<T>& frame,
                                                 const Matrix
                                                 <unsigned char>& filter,
                                                 Morphology type)
{
    if (filter.size() <= 1)
        return frame;

    const unsigned int width = frame.cols();
    const unsigned int height = frame.rows();
    const unsigned int size = width * height;

    Matrix<T> result(height, width);

    const int fCenterX = filter.cols() / 2;
    const int fCenterY = filter.rows() / 2;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 256)
#else
#pragma omp parallel for if (height > 16 && size > 256)
#endif
    for (int y = 0; y < (int)height; ++y) {
        for (int x = 0; x < (int)width; ++x) {
            T val = (type == Erode) ? std::numeric_limits<T>::max()
                                    : std::numeric_limits<T>::min();

            for (unsigned int fy = 0, fyMax = filter.rows(); fy < fyMax; ++fy) {
                for (unsigned int fx = 0, fxMax = filter.cols(); fx < fxMax;
                     ++fx) {
                    const int mx = x + (fx - fCenterX);
                    const int my = y + (fy - fCenterY);

                    if (mx >= 0 && mx < (int)width && my >= 0
                        && my < (int)height) {
                        val = (type == Erode) ? std::min(val, frame(my, mx))
                                              : std::max(val, frame(my, mx));
                    }
                }
            }

            result(y, x) = val;
        }
    }

    return result;
}

#endif // N2D2_COMPUTERVISION_MORPHOLOGY_H
