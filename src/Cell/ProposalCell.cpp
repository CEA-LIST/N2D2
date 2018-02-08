/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Cell/ProposalCell.hpp"

const char* N2D2::ProposalCell::Type = "Proposal";

N2D2::ProposalCell::ProposalCell(const std::string& name,
                                 StimuliProvider& sp,
                                 const unsigned int nbOutputs,
                                 unsigned int nbProposals,
                                 unsigned int scoreIndex,
                                 unsigned int IoUIndex,
                                 bool isNMS,
                                 std::vector<double> meanFactor,
                                 std::vector<double> stdFactor,
                                 std::vector<unsigned int> numParts,
                                 std::vector<unsigned int> numTemplates)
    : Cell(name, nbOutputs),
      mNMS_IoU_Threshold(this, "NMS_IoU_Threshold", 0.3),
      mScoreThreshold(this, "Score_Threshold", 0.0),
      mKeepMax(this, "KeepMaxCls", false),
      mStimuliProvider(sp),      
      mNbProposals(nbProposals),
      mScoreIndex(scoreIndex),
      mIoUIndex(IoUIndex),
      mApplyNMS(isNMS),
      mMeanFactor(meanFactor),
      mStdFactor(stdFactor),
      mNumParts(numParts),
      mNumTemplates(numTemplates)
{

}

void N2D2::ProposalCell::getStats(Stats& /*stats*/) const
{

}

void N2D2::ProposalCell::setOutputsSize()
{
    mOutputsWidth = 1;
    mOutputsHeight = 1;
}
