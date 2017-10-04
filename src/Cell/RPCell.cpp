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

#include "Cell/RPCell.hpp"

const char* N2D2::RPCell::Type = "RP";

N2D2::RPCell::RPCell(const std::string& name,
                     unsigned int nbAnchors,
                     unsigned int nbProposals,
                     unsigned int scoreIndex,
                     unsigned int IoUIndex)
    : Cell(name, 4),
      mMinWidth(this, "MinWidth", 0.0),
      mMinHeight(this, "MinHeight", 0.0),
      mNMS_IoU_Threshold(this, "NMS_IoU_Threshold", 0.7),
      mPre_NMS_TopN(this, "Pre_NMS_TopN", 0U),
      mForegroundRate(this, "ForegroundRate", 0.25),
      mForegroundMinIoU(this, "ForegroundMinIoU", 0.5),
      mBackgroundMaxIoU(this, "BackgroundMaxIoU", 0.5),
      mBackgroundMinIoU(this, "BackgroundMinIoU", 0.1),
      mNbAnchors(nbAnchors),
      mNbProposals(nbProposals),
      mScoreIndex(scoreIndex),
      mIoUIndex(IoUIndex)
{
    // ctor
}

void N2D2::RPCell::getStats(Stats& /*stats*/) const
{

}

void N2D2::RPCell::setOutputsSize()
{
    mOutputsWidth = 1;
    mOutputsHeight = 1;
}
