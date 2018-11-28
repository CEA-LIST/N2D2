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

#ifndef N2D2_ACTIVATION_KERNELS_H
#define N2D2_ACTIVATION_KERNELS_H

#include "utils/Utils.hpp"
#include "third_party/half.hpp"

#include "Activation/Activation.hpp"
#include "Solver/SGDSolver_Kernels.hpp"

namespace N2D2 {
void rangeAveraging(double minVal,
                    double maxVal,
                    double& minAveragedVal,
                    double& maxAveragedVal,
                    unsigned long long int& nbSteps,
                    Activation::MovingAverageType movingAverage,
                    unsigned int MA_Window,
                    double EMA_Alpha);

double log2Round(double value, double rate = 1.0, double power = 0.0);
}

#endif // N2D2_ACTIVATION_KERNELS_H
