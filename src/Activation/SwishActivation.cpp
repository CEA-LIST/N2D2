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

#include "Activation/SwishActivation.hpp"

const char* N2D2::SwishActivation::Type = "Swish";

N2D2::SwishActivation::SwishActivation()
    : mMaxValMA(0.0),
      mMaxValQuant(0.0)
{
    // ctor
}

std::pair<double, double> N2D2::SwishActivation::getOutputRange() const {
    const double max = std::numeric_limits<double>::infinity();
    return std::make_pair(-max, max);
}

void N2D2::SwishActivation::saveInternal(std::ostream& state,
                                             std::ostream& log) const
{
    state.write(reinterpret_cast<const char*>(&mMaxValMA), sizeof(mMaxValMA));
    state.write(reinterpret_cast<const char*>(&mMaxValQuant),
                sizeof(mMaxValQuant));

    log << "Range after moving average (*MA): [0, " << mMaxValMA << "]\n"
        << "Quantization range (*Quant): [0, " << mMaxValQuant << "]\n"
        << "Quantization range after rescaling: [0, "
            << (mMaxValQuant / mPreQuantizeScaling) << "]" << std::endl;
}

void N2D2::SwishActivation::loadInternal(std::istream& state)
{
    state.read(reinterpret_cast<char*>(&mMaxValMA), sizeof(mMaxValMA));
    state.read(reinterpret_cast<char*>(&mMaxValQuant), sizeof(mMaxValQuant));
}
