/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_SCALING_MODE_H
#define N2D2_SCALING_MODE_H

#include <stdexcept>
#include <string>

enum class ScalingMode {
    NONE,
    FLOAT_MULT,
    FIXED_MULT,
    SINGLE_SHIFT,
    DOUBLE_SHIFT,
};

inline ScalingMode parseScalingMode(const std::string& str) {
    if(str == "Floating-point") {
        return ScalingMode::FLOAT_MULT;
    }

    if(str == "Fixed-point") {
        return ScalingMode::FIXED_MULT;
    }

    if(str == "Single-shift") {
        return ScalingMode::SINGLE_SHIFT;
    }

    if(str == "Double-shift") {
        return ScalingMode::DOUBLE_SHIFT;
    }

    throw std::runtime_error("Unknown scaling mode '" + str + "'.");
}

#endif