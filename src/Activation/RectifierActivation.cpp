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

#include "Activation/RectifierActivation.hpp"

const char* N2D2::RectifierActivation::Type = "Rectifier";

N2D2::RectifierActivation::RectifierActivation()
    : mLeakSlope(this, "LeakSlope", 0.0),
      mClipping(this, "Clipping", 0.0)
{
    // ctor
}

std::pair<double, double> N2D2::RectifierActivation::getOutputRange() const {
    const double max = mClipping > 0.0?mClipping:std::numeric_limits<double>::infinity();
    if(mLeakSlope > 0.0) {
        return std::make_pair(-max, max);
    }
    else {
        return std::make_pair(0.0, max);
    }
}
