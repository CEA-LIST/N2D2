/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#include "Cell/PaddingCell.hpp"
#include "DeepNet.hpp"

const char* N2D2::PaddingCell::Type = "Padding";

N2D2::PaddingCell::PaddingCell(const DeepNet& deepNet, const std::string& name,
                               unsigned int nbOutputs,
                               int topPad,
                               int botPad,
                               int leftPad,
                               int rightPad)
    : Cell(deepNet, name,
            nbOutputs),
      mTopPad(topPad),
      mBotPad(botPad),
      mLeftPad(leftPad),
      mRightPad(rightPad)
{
    // ctor
}

std::vector<unsigned int> N2D2::PaddingCell::getReceptiveField(
    const std::vector<unsigned int>& outputField) const
{
    return outputField;
}

void N2D2::PaddingCell::getStats(Stats& stats) const
{
    stats.nbNodes += getOutputsSize();
}

void N2D2::PaddingCell::setOutputsDims()
{
    mOutputsDims[0] = (mInputsDims[0] + mLeftPad + mRightPad);
    mOutputsDims[1] = (mInputsDims[1] + mTopPad + mBotPad);
}

std::pair<double, double> N2D2::PaddingCell::getOutputsRange() const {
    std::pair<double, double> parentOutputsRange = Cell::getOutputsRangeParents();

    if(parentOutputsRange.first > 0.0)
        parentOutputsRange.first = 0.0;

    return parentOutputsRange;
}
