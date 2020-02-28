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
#include "Cell/TargetBiasCell.hpp"
#include "utils/Registrar.hpp"

const char* N2D2::TargetBiasCell::Type = "TargetBias";

N2D2::RegistryMap_T& N2D2::TargetBiasCell::registry() {
    static RegistryMap_T rMap;
    return rMap;
}

N2D2::TargetBiasCell::TargetBiasCell(const DeepNet& deepNet, const std::string& name,
                               unsigned int nbOutputs, double bias)
    : Cell(deepNet, name, nbOutputs),
      mBias(std::move(bias))
{
}

const char* N2D2::TargetBiasCell::getType() const {
    return Type;
}

void N2D2::TargetBiasCell::getStats(Stats& /*stats*/) const {

}

void N2D2::TargetBiasCell::setOutputsDims() {
    mOutputsDims[0] = mInputsDims[0];
    mOutputsDims[1] = mInputsDims[1];
}
