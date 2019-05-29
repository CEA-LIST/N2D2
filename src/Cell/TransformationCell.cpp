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

#include "Cell/TransformationCell.hpp"
#include "DeepNet.hpp"
#include "Transformation/Transformation.hpp"

const char* N2D2::TransformationCell::Type = "Transformation";

N2D2::TransformationCell::TransformationCell(const DeepNet& deepNet, const std::string& name,
                                             unsigned int nbOutputs,
                                             const std::shared_ptr
                                             <Transformation>& transformation)
    : Cell(deepNet, name, nbOutputs), mTransformation(transformation)
{
    // ctor
}

void N2D2::TransformationCell::getStats(Stats& /*stats*/) const
{
}

void N2D2::TransformationCell::setOutputsDims()
{
    unsigned int outputsWidth, outputsHeight;
    std::tie(outputsWidth, outputsHeight)
        = mTransformation->getOutputsSize(mInputsDims[0], mInputsDims[1]);

    if (outputsWidth == 0 && outputsHeight == 0)
        throw std::runtime_error(
            "TransformationCell outputs size must be known in advance");

    mOutputsDims[0] = outputsWidth;
    mOutputsDims[1] = outputsHeight;
}
