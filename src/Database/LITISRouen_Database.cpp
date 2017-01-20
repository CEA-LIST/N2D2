/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "Database/LITISRouen_Database.hpp"

N2D2::LITISRouen_Database::LITISRouen_Database(double learn, double validation)
    : DIR_Database(), mLearn(learn), mValidation(validation)
{
    // ctor
}

void N2D2::LITISRouen_Database::load(const std::string& dataPath,
                                     const std::string& /*labelPath*/,
                                     bool /*extractROIs*/)
{
    loadDir(dataPath, 0, "", -1);

    for (std::vector<Stimulus>::iterator it = mStimuli.begin(),
                                         itEnd = mStimuli.end();
         it != itEnd;
         ++it) {
        std::string labelName = Utils::baseName((*it).name);
        labelName = labelName.substr(0, labelName.find_first_of("0123456789"));

        (*it).label = labelID(labelName);
    }

    partitionStimuli(mLearn, mValidation, 1.0 - mLearn - mValidation);
}
