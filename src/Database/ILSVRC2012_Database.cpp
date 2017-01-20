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

#include "Database/ILSVRC2012_Database.hpp"

N2D2::ILSVRC2012_Database::ILSVRC2012_Database(double learn,
                                               bool useValidationForTest)
    : DIR_Database(), mLearn(learn), mUseValidationForTest(useValidationForTest)
{
    // ctor
}

void N2D2::ILSVRC2012_Database::load(const std::string& dataPath,
                                     const std::string& labelPath,
                                     bool /*extractROIs*/)
{
    // Learn & Tests Stimuli
    loadImageNetStimuliPerDir(dataPath + "/train", labelPath);
    partitionStimuli(mLearn, 0, 1.0 - mLearn);

    // Validation stimuli
    loadImageNetValidationStimuli(dataPath + "/val", dataPath + "/val.txt");
    partitionStimuli(0.0, 1.0, 0.0);

    if (mUseValidationForTest) {
        // Test stimuli (using validation database)
        loadImageNetValidationStimuli(dataPath + "/val", dataPath + "/val.txt");
        partitionStimuli(0.0, 0.0, 1.0);
    }
}

void N2D2::ILSVRC2012_Database::loadImageNetStimuliPerDir(const std::string
                                                          & dirPath,
                                                          const std::string
                                                          & labelNamePath)
{
    std::ifstream labels(labelNamePath.c_str());
    if (!labels.good())
        throw std::runtime_error("Could not open labels file: "
                                 + labelNamePath);

    std::string classDir;

    while (labels >> classDir)
        loadDir(dirPath + "/" + classDir, 0, classDir, 0);
}

void N2D2::ILSVRC2012_Database::loadImageNetValidationStimuli(const std::string
                                                              & dirPath,
                                                              const std::string
                                                              & valNamePath)
{
    std::ifstream validationFile(valNamePath.c_str());
    if (!validationFile.good())
        throw std::runtime_error("Could not open validation labels file: "
                                 + valNamePath);

    std::string stimuliName;
    unsigned int labelID;

    while (validationFile >> stimuliName >> labelID)
        loadFile(dirPath + "/" + stimuliName, getLabelName(labelID));
}
