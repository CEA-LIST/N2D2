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

#include "Cell/LRNCell.hpp"

const char* N2D2::LRNCell::Type = "LRN";

N2D2::LRNCell::LRNCell(const std::string& name, unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      mN(this, "N", 5U),
      mAlpha(this, "Alpha", 1.0e-4),
      mBeta(this, "Beta", 0.75),
      mK(this, "K", 2.0)
{
    // ctor
}

void N2D2::LRNCell::getStats(Stats& stats) const
{
    stats.nbNodes += getNbOutputs() * getOutputsWidth() * getOutputsHeight();
}

void N2D2::LRNCell::setOutputsSize()
{
    mOutputsWidth = mChannelsWidth;
    mOutputsHeight = mChannelsHeight;
}
