/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "Activation/Activation.hpp"
#include "Cell/ActivationCell.hpp"
#include "DeepNet.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"
#include "StimuliProvider.hpp"

const char* N2D2::ActivationCell::Type = "Activation";

N2D2::ActivationCell::ActivationCell(const DeepNet& deepNet, 
                         const std::string& name,
                         unsigned int nbOutputs)
    : Cell(deepNet, name, nbOutputs)
{
    // ctor
}

std::vector<unsigned int> N2D2::ActivationCell::getReceptiveField(
    const std::vector<unsigned int>& outputField) const
{
    return outputField;
}

void N2D2::ActivationCell::getStats(Stats& stats) const
{
    stats.nbNodes += getOutputsSize();
}

void N2D2::ActivationCell::setOutputsDims()
{
    mOutputsDims[0] = mInputsDims[0];
    mOutputsDims[1] = mInputsDims[1];
}
