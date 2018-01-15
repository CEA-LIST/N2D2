/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND(david.briand@cea.fr)

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

#include "Cell/ROIPoolingCell.hpp"

const char* N2D2::ROIPoolingCell::Type = "ROIPooling";

N2D2::ROIPoolingCell::ROIPoolingCell(const std::string& name,
                                     StimuliProvider& sp,
                                     unsigned int outputsWidth,
                                     unsigned int outputsHeight,
                                     unsigned int nbOutputs,
                                     ROIPooling pooling)
    : Cell(name, nbOutputs),
      mFlip(this, "Flip", false),
      mIgnorePad(this, "IgnorePadding", 0),
      mStimuliProvider(sp),
      mPooling(pooling)
{
    // ctor
    mOutputsWidth = outputsWidth;
    mOutputsHeight = outputsHeight;
}

void N2D2::ROIPoolingCell::getStats(Stats& stats) const
{
    stats.nbNodes += getNbOutputs() * getOutputsWidth() * getOutputsHeight();
}

void N2D2::ROIPoolingCell::setInputsSize(unsigned int width,
                                         unsigned int height)
{
    mChannelsWidth = width;
    mChannelsHeight = height;
}

void N2D2::ROIPoolingCell::setOutputsSize()
{

}
