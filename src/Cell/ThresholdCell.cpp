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

#include "Cell/ThresholdCell.hpp"

const char* N2D2::ThresholdCell::Type = "Threshold";

N2D2::ThresholdCell::ThresholdCell(const std::string& name,
                                   unsigned int nbOutputs,
                                   double threshold)
    : Cell(name, nbOutputs),
      mThreshold(threshold),
      mOperation(this, "Operation", Binary),
      mMaxValue(this, "MaxValue", 1.0)
{
    // ctor
}

void N2D2::ThresholdCell::getStats(Stats& stats) const
{
    stats.nbNodes += getOutputsSize();
}

void N2D2::ThresholdCell::setOutputsDims()
{
    mOutputsDims = mInputsDims;
}
