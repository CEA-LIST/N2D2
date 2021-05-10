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

#include "Cell/ElemWiseCell.hpp"
#include "DeepNet.hpp"

const char* N2D2::ElemWiseCell::Type = "ElemWise";

N2D2::ElemWiseCell::ElemWiseCell(const DeepNet& deepNet, const std::string& name,
                         unsigned int nbOutputs,
                         Operation operation,
                         CoeffMode mode,
                         const std::vector<Float_T>& weights,
                         const std::vector<Float_T>& shifts)
    : Cell(deepNet, name, nbOutputs),
      mOperation(operation),
      mCoeffMode(mode),
      mWeights(weights),
      mShifts(shifts)
{
    // ctor
}

std::vector<unsigned int> N2D2::ElemWiseCell::getReceptiveField(
    const std::vector<unsigned int>& outputField) const
{
    return outputField;
}

void N2D2::ElemWiseCell::getStats(Stats& stats) const
{
    stats.nbNodes += getOutputsSize();
}

void N2D2::ElemWiseCell::setOutputsDims()
{
    mOutputsDims[1] = mInputsDims[1];
    mOutputsDims[0] = mInputsDims[0];

}

std::pair<double, double> N2D2::ElemWiseCell::getOutputsRange() const {
    const std::pair<double, double> parentOutputsRange = Cell::getOutputsRangeParents();

    const double inf = std::numeric_limits<double>::infinity();
    if(mOperation == EuclideanSum) {
        return std::make_pair(0, inf);
    }

    if(parentOutputsRange.first >= 0.0) {
        if(mOperation == Sum && 
           std::all_of(mShifts.begin(), mShifts.end(), [](Float_T v) { return v >= 0.0; }) && 
           std::all_of(mWeights.begin(), mWeights.end(), [](Float_T v) { return v >= 0.0; }))
        {
            return std::make_pair(0, inf);
        }
        else if(mOperation == AbsSum && 
                std::all_of(mWeights.begin(), mWeights.end(), [](Float_T v) { return v >= 0.0; }))
        {
            return std::make_pair(0, inf);
        }
        else if(mOperation == Prod) {
            return std::make_pair(0, inf);
        }
        else if(mOperation == Max) {
            return std::make_pair(0, inf);
        }
    }

    return std::make_pair(-inf, inf);
}
