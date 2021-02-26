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
#include "Cell/TransposeCell.hpp"

const char* N2D2::TransposeCell::Type = "Transpose";

N2D2::TransposeCell::TransposeCell(const DeepNet& deepNet, 
                         const std::string& name,
                         unsigned int nbOutputs,
                         const std::vector<int>& perm)
    : Cell(deepNet, name, nbOutputs),
      mPerm(perm)
{
    // ctor
    const int size = mPerm.size();

    if (size > 4) {
        throw std::domain_error("TransposeCell: the number of dimensions for"
                                " the permutation cannot be above 4.");
    }

    if (size < 4) {
        mPerm.resize(4);
        std::iota(mPerm.begin() + size, mPerm.end(), size);
    }

    // Permutation of the fourth dim not allowed
    if (mPerm.back() != 3) {
        throw std::domain_error("TransposeCell: "
                                "permutation of the fourth (batch) dimension "
                                "is not supported.");
    }

    // Check that the perm vector is indeed a valid permutation
    std::vector<int> noPerm(mPerm.size());
    std::iota(noPerm.begin(), noPerm.end(), 0);

    if (!std::is_permutation(mPerm.begin(), mPerm.end(), noPerm.begin())) {
        throw std::domain_error("TransposeCell: the perm argument is not a"
                                " permutation!");
    }
}

std::vector<unsigned int> N2D2::TransposeCell::getReceptiveField(
    const std::vector<unsigned int>& outputField) const
{
    return outputField;
}

void N2D2::TransposeCell::getStats(Stats& /*stats*/) const
{
    
}

void N2D2::TransposeCell::setOutputsDims()
{
    mOutputsDims.resize(mInputsDims.size());

    // Permutation of the fourth dim not allowed, the fourth dim is not used.
    for (unsigned int dim = 0; dim < mInputsDims.size(); ++dim)
        mOutputsDims[dim] = mInputsDims[mPerm[dim]];
}

std::vector<int> N2D2::TransposeCell::getInversePermutation() const
{
    std::vector<int> invPerm(mPerm.size());

    for (unsigned int dim = 0; dim < mPerm.size(); ++dim)
        invPerm[mPerm[dim]] = dim;

    return invPerm;
}
