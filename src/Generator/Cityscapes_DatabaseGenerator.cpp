/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#ifdef JSONCPP

#include "Generator/Cityscapes_DatabaseGenerator.hpp"

N2D2::Registrar<N2D2::DatabaseGenerator>
N2D2::Cityscapes_DatabaseGenerator::mRegistrar(
    "Cityscapes_Database", N2D2::Cityscapes_DatabaseGenerator::generate);

std::shared_ptr<N2D2::Cityscapes_Database>
N2D2::Cityscapes_DatabaseGenerator::generate(IniParser& iniConfig,
                                           const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const bool incTrainExtra = iniConfig.getProperty<bool>("IncTrainExtra",
                                                           false);
    const bool useCoarse = iniConfig.getProperty<bool>("UseCoarse", false);
    const bool singleInstanceLabels = iniConfig.getProperty<bool>
        ("SingleInstanceLabels", true);

    std::string defaultPath = N2D2_DATA("Cityscapes/leftImg8bit");
    const char* cityscapesRoot = std::getenv("CITYSCAPES_DATASET");

    if (cityscapesRoot != NULL)
        defaultPath = std::string(cityscapesRoot);

    const std::string dataPath
        = Utils::expandEnvVars(iniConfig.getProperty<std::string>(
            "DataPath", defaultPath));
    const std::string labelPath
        = Utils::expandEnvVars(iniConfig.getProperty<std::string>("LabelPath",
                                                                  ""));

    std::shared_ptr<Cityscapes_Database> database = std::make_shared
        <Cityscapes_Database>(incTrainExtra, useCoarse, singleInstanceLabels);
    database->setParameters(iniConfig.getSection(section, true));
    database->load(dataPath, labelPath);
    return database;
}

#endif
