/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#include "Cell/ObjectDetCell.hpp"

const char* N2D2::ObjectDetCell::Type = "ObjectDet";

N2D2::ObjectDetCell::ObjectDetCell(const std::string& name,
                                 StimuliProvider& sp,
                                const unsigned int nbOutputs,
                                unsigned int nbAnchors,
                                unsigned int nbProposals,
                                unsigned int nbClass,
                                Float_T nmsThreshold,
                                Float_T scoreThreshold)
    : Cell(name, nbOutputs),
      mStimuliProvider(sp), 
      mNbAnchors(nbAnchors),     
      mNbProposals(nbProposals),
      mNbClass(nbClass),
      mNMS_IoU_Threshold(nmsThreshold),
      mScoreThreshold(scoreThreshold)
{
}

void N2D2::ObjectDetCell::getStats(Stats& /*stats*/) const
{

}

void N2D2::ObjectDetCell::setOutputsDims()
{
    mOutputsDims[0] = 1;
    mOutputsDims[1] = 1;
}
