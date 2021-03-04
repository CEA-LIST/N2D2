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

#include "DeepNet.hpp"
#include "Cell/ReshapeCell.hpp"

const char* N2D2::ReshapeCell::Type = "Reshape";

N2D2::ReshapeCell::ReshapeCell(const DeepNet& deepNet, 
                         const std::string& name,
                         unsigned int nbOutputs,
                         const std::vector<int>& dims)
    : Cell(deepNet, name, nbOutputs),
      mDims(dims)
{
    // ctor
    const int size = mDims.size();

    if (size > 4) {
        throw std::domain_error("ReshapeCell: the number of dimensions for"
                                " the reshape cannot be above 4.");
    }
    else if (size > 3 && mDims[3] != 0) {
        std::cout << Utils::cwarning << "ReshapeCell: fourth (batch) dimension"
            " will be ignored." << Utils::cdef << std::endl;
    }

    mDims.resize(3, 1);
}

std::vector<unsigned int> N2D2::ReshapeCell::getReceptiveField(
    const std::vector<unsigned int>& outputField) const
{
    return outputField;
}

void N2D2::ReshapeCell::getStats(Stats& /*stats*/) const
{
    
}

void N2D2::ReshapeCell::setOutputsDims()
{
    const size_t inputsSize = getInputsSize();

    std::vector<int> dims(mDims);
    int inferDimIndex = -1;
    size_t size = 1;

    for (size_t dim = 0; dim < dims.size(); ++dim) {
        if (dims[dim] == 0)
            dims[dim] = mInputsDims[dim];

        if (dims[dim] == -1) {
            if (inferDimIndex >= 0) {
                throw std::domain_error("ReshapeCell: at most one dimension "
                    "of the new shape can be -1.");
            }
            else
                inferDimIndex = dim;
        }
        else
            size *= dims[dim];
    }

    if (inferDimIndex >= 0 && size > 0)
        dims[inferDimIndex] = inputsSize / size;

    const size_t outputsSize = std::accumulate(dims.begin(), dims.end(),
                                               1U, std::multiplies<size_t>());

    if (inputsSize != outputsSize) {
        std::stringstream msgStr;
        msgStr << "ReshapeCell: the total size of the first 3 dimensions after"
            " reshape (" << dims << ") doesn't match the corresponding total"
            " input size (" << mInputsDims << ")" << std::endl;

        throw std::domain_error(msgStr.str());
    }

    mOutputsDims = std::vector<size_t>(dims.begin(), dims.end());
}
