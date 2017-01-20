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

#include "Database/GTSRB_DIR_Database.hpp"

N2D2::GTSRB_DIR_Database::GTSRB_DIR_Database(double validation)
    : DIR_Database(), mValidation(validation)
{
    // ctor
    mValidExtensions.push_back("ppm");
}

void N2D2::GTSRB_DIR_Database::load(const std::string& dataPath,
                                    const std::string& labelPath,
                                    bool extractROIs)
{
    const std::string labelPathDef = (labelPath.empty()) ? dataPath : labelPath;

    // Learn and validation stimuli
    loadDir(dataPath + "/Final_Training/Images", 1, "", -1);
    loadROIsDir(labelPathDef + "/Final_Training/Images", "csv", 1);

    if (extractROIs)
        Database::extractROIs();
    else
        extractLabels();

    partitionStimuliPerLabel(1.0 - mValidation, mValidation, 0.0);

    // Test stimuli
    loadDir(dataPath + "/Final_Test/Images", 0, "", -1);
    loadROIs(labelPathDef + "/GT-final_test.csv",
             dataPath + "/Final_Test/Images");

    if (extractROIs)
        Database::extractROIs();
    else
        extractLabels();

    partitionStimuli(0.0, 0.0, 1.0);
}
