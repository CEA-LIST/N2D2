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

#include <algorithm>
#include <limits>
#include <string>

#include "DeepNet.hpp"
#include "Scaling.hpp"
#include "Cell/ScalingCell.hpp"
#include "utils/Registrar.hpp"

const char* N2D2::ScalingCell::Type = "Scaling";

N2D2::RegistryMap_T& N2D2::ScalingCell::registry() {
    static RegistryMap_T rMap;
    return rMap;
}

N2D2::ScalingCell::ScalingCell(const DeepNet& deepNet, const std::string& name,
                               unsigned int nbOutputs, Scaling scaling)
    : Cell(deepNet, name, nbOutputs),
      mScaling(scaling)
{
}

const char* N2D2::ScalingCell::getType() const {
    return Type;
}

void N2D2::ScalingCell::getStats(Stats& /*stats*/) const {
}

const N2D2::Scaling& N2D2::ScalingCell::getScaling() const {
    return mScaling;
}

N2D2::Scaling& N2D2::ScalingCell::getScaling() {
    return mScaling;
}

void N2D2::ScalingCell::setScaling(Scaling scaling) {
    mScaling = scaling;
}

void N2D2::ScalingCell::setOutputsDims() {
    mOutputsDims[0] = mInputsDims[0];
    mOutputsDims[1] = mInputsDims[1];
}

std::pair<double, double> N2D2::ScalingCell::getOutputsRange() const {
    bool allScalingsPositive = false;
    if(mScaling.getMode() == ScalingMode::FLOAT_MULT) {
        const std::vector<Float_T>& scalingPerOutput = mScaling.getFloatingPointScaling()
                                                               .getScalingPerOutput();

        allScalingsPositive = std::all_of(scalingPerOutput.begin(), scalingPerOutput.end(), 
                                          [](Float_T v) { return v >= 0.0; });
    }
    else if(mScaling.getMode() == ScalingMode::FIXED_MULT) {
        const std::vector<std::int32_t>& scalingPerOutput = mScaling.getFixedPointScaling()
                                                                    .getScalingPerOutput();

        allScalingsPositive = std::all_of(scalingPerOutput.begin(), scalingPerOutput.end(), 
                                          [](std::int32_t v) { return v >= 0; });
    }
    else if(mScaling.getMode() == ScalingMode::SINGLE_SHIFT || 
            mScaling.getMode() == ScalingMode::SINGLE_SHIFT) 
    {
        allScalingsPositive = true;
    }
    else {
        throw std::runtime_error("Unsupported scaling mode for cell " + getName() + ".");
    }


    const double inf = std::numeric_limits<double>::infinity();
    
    const std::pair<double, double> parentOutputsRange = Cell::getOutputsRangeParents();
    if(parentOutputsRange.first >= 0.0 && allScalingsPositive) {
        return std::make_pair(0, inf);
    }
    else {
        return std::make_pair(-inf, inf);
    }
}