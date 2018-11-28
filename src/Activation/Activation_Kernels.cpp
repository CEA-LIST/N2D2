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

#include "Activation/Activation_Kernels.hpp"

void N2D2::rangeAveraging(double minVal,
                          double maxVal,
                          double& minAveragedVal,
                          double& maxAveragedVal,
                          unsigned long long int& nbSteps,
                          Activation::MovingAverageType movingAverage,
                          unsigned int MA_Window,
                          double EMA_Alpha)
{
    if (nbSteps == 0) {
        minAveragedVal = minVal;
        maxAveragedVal = maxVal;
    }
    else {
        if (movingAverage == Activation::EMA) {
            const double alpha = (EMA_Alpha != 0.0)
                ? EMA_Alpha
                : 2.0 / (MA_Window + 1);

            minAveragedVal = alpha * minVal
                        + (1.0 - alpha) * minAveragedVal;
            maxAveragedVal = alpha * maxVal
                        + (1.0 - alpha) * maxAveragedVal;
        }
        else {
            throw std::domain_error("rangeAveraging(): only EMA moving average "
                                    "is currently supported.");
        }
    }

    ++nbSteps;
}

double N2D2::log2Round(double value, double rate, double power)
{
    assert(rate >= 0.0 && rate <= 1.0);
    assert(power >= 0.0);

    if (value == 0.0)
        return 0.0;

    const double sign = (value >= 0.0) ? 1.0 : -1.0;
    const double log2Value = log2(std::abs(value));
    const double log2Up = std::ceil(log2Value);
    const double log2Down = std::floor(log2Value);
    const double log2Middle = (log2Up + log2Down) / 2.0;

    const double a = (log2Up - log2Down) / 2.0;
    const double b = - a * log2Middle;
    const double x = a * log2Value + b;
    const double corr = rate * std::pow(std::abs(x), power);

    const double log2Target
        = (std::abs(log2Down - log2Value) < std::abs(log2Up - log2Value))
            ? log2Down : log2Up;

    return sign * std::pow(2.0, (1.0 - corr) * log2Value + corr * log2Target);
}
