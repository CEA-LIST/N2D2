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

#include "Cell/AnchorCell.hpp"

const char* N2D2::AnchorCell::Type = "Anchor";

N2D2::AnchorCell::AnchorCell(
    const std::string& name,
    StimuliProvider& sp,
    const std::vector<AnchorCell_Frame_Kernels::Anchor>& anchors,
    unsigned int scoresCls)
    : Cell(name, 6*anchors.size()),
      mPositiveIoU(this, "PositiveIoU", 0.7),
      mNegativeIoU(this, "NegativeIoU", 0.3),
      mLossLambda(this, "LossLambda", 10.0),
      mLossPositiveSample(this, "LossPositiveSample", 128U),
      mLossNegativeSample(this, "LossNegativeSample", 128U),
      mFlip(this, "Flip", false),
      mStimuliProvider(sp),
      mScoresCls(scoresCls)
{
    // ctor
}

void N2D2::AnchorCell::getStats(Stats& /*stats*/) const
{

}

void N2D2::AnchorCell::setOutputsSize()
{
    mOutputsWidth = mChannelsWidth;
    mOutputsHeight = mChannelsHeight;
}
