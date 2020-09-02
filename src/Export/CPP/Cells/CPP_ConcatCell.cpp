/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <stdexcept>
#include <string>

#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame.hpp"
#include "DeepNet.hpp"
#include "Export/CPP/Cells/CPP_ConcatCell.hpp"

const char* N2D2::CPP_ConcatCell::Type = "Concat";

N2D2::RegistryMap_T& N2D2::CPP_ConcatCell::registry() {
    static RegistryMap_T rMap;
    return rMap;
}

N2D2::CPP_ConcatCell::CPP_ConcatCell(const DeepNet& deepNet, const std::string& name, 
                                                 unsigned int nbOutputs): 
        Cell(deepNet, name, nbOutputs)
{
}

const char* N2D2::CPP_ConcatCell::getType() const {
    return Type;
}

void N2D2::CPP_ConcatCell::getStats(Stats& /*stats*/) const {
}

void N2D2::CPP_ConcatCell::setOutputsDims() {
    mOutputsDims[0] = mInputsDims[0];
    mOutputsDims[1] = mInputsDims[1];
}

std::pair<double, double> N2D2::CPP_ConcatCell::getOutputsRange() const {
    return getOutputsRangeParents();
}
