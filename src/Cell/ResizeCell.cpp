/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND(david.briand@cea.fr)

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

#include "Cell/ResizeCell.hpp"

const char* N2D2::ResizeCell::Type = "Resize";

N2D2::ResizeCell::ResizeCell(const std::string& name,
                             unsigned int outputsWidth,
                             unsigned int outputsHeight,
                             unsigned int nbOutputs,
                             ResizeMode resizeMode)
    : Cell(name, nbOutputs),
      mAlignCorners(this, "AlignCorners", true),
      mResizeMode(resizeMode),
      mResizeOutputWidth(outputsWidth),
      mResizeOutputHeight(outputsHeight)
{
    // ctor
}

void N2D2::ResizeCell::getStats(Stats& stats) const
{
    stats.nbNodes += getNbOutputs() * getOutputsWidth() * getOutputsHeight();
}
/*
void N2D2::ResizeCell::setInputsDims(const std::vector<size_t>& dims)
{
    if (mInputsDims.empty())
        mInputsDims = dims;
    else {
        mInputsDims[0] = dims[0];
        mInputsDims[1] = dims[1];
        mInputsDims.back() += dims.back();
    }
}
*/

void N2D2::ResizeCell::setOutputsDims()
{
    mOutputsDims[0] = mResizeOutputWidth;
    mOutputsDims[1] = mResizeOutputHeight;

}