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

#ifndef N2D2_SGDSOLVER_KERNELS_H
#define N2D2_SGDSOLVER_KERNELS_H

#include "containers/Tensor.hpp"
#include "utils/Utils.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
template <class T>
std::pair<T, T> minMax(const Tensor<T>& x);

template <class T>
void rangeZeroAlign(T minVal,
                    T maxVal,
                    T& minAlignedVal,
                    T& maxAlignedVal,
                    unsigned int quantizationLevels,
                    bool zeroPointFree = true);

template <class T>
void quantize(Tensor<T>& y,
              const Tensor<T>& x,
              T minVal,
              T maxVal,
              unsigned int quantizationLevels,
              bool truncate = false);
}

template <class T>
std::pair<T, T> N2D2::minMax(const Tensor<T>& x)
{
    // Compute global min & max value on the full tensor
    const std::pair<typename Tensor<T>::const_iterator,
              typename Tensor<T>::const_iterator> minMaxPair
        = std::minmax_element(x.begin(), x.end());
    return std::make_pair(*(minMaxPair.first), *(minMaxPair.second));
}

template <class T>
void N2D2::rangeZeroAlign(T minVal,
                          T maxVal,
                          T& minAlignedVal,
                          T& maxAlignedVal,
                          unsigned int quantizationLevels,
                          bool zeroPointFree)
{
    if (quantizationLevels > 1) {
        using namespace std;
        using namespace half_float;

        // absMaxVal is used to perform symmetric zero-point-free quantization
        const T absMaxVal = fmax(fabs(minVal), fabs(maxVal));

        std::tie(minAlignedVal, maxAlignedVal)
            = Utils::zeroAlignedQuantizedRange<T>(
                (zeroPointFree) ? -absMaxVal : minVal,
                (zeroPointFree) ? absMaxVal : maxVal,
                quantizationLevels);
    }
    else {
        minAlignedVal = -1.0;
        maxAlignedVal = 1.0;
    }
}

template <class T>
void N2D2::quantize(Tensor<T>& y,
                    const Tensor<T>& x,
                    T minVal,
                    T maxVal,
                    unsigned int quantizationLevels,
                    bool truncate)
{
    using namespace std;
    using namespace half_float;

    if (quantizationLevels > 1) {
        const T scaling = (maxVal - minVal) / (T)(quantizationLevels - 1);

#pragma omp parallel for if (x.size() > 1024)
        for (int i = 0; i < (int)x.size(); ++i) {
            const T clamped = (x(i) < minVal) ? minVal :
                              (x(i) > maxVal) ? maxVal :
                                                x(i);

            if (truncate)
                y(i) = (int)((clamped - minVal) / scaling) * scaling + minVal;
            else {
                y(i) = (int)Utils::round<T>((clamped - minVal) / scaling)
                                * scaling + minVal;
            }
        }
    }
    else {
#pragma omp parallel for if (x.size() > 1024)
        for (int i = 0; i < (int)x.size(); ++i) {
            y(i) = ((x(i) >= 0.0) ? 1.0 : -1.0);
        }
    }
}

#endif // N2D2_SGDSOLVER_KERNELS_H
