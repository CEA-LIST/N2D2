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

#include "Generator/DIR_DatabaseGenerator.hpp"

N2D2::Registrar<N2D2::DatabaseGenerator>
N2D2::DIR_DatabaseGenerator::mRegistrar("DIR_Database",
                                        N2D2::DIR_DatabaseGenerator::generate);

std::shared_ptr<N2D2::DIR_Database>
N2D2::DIR_DatabaseGenerator::generate(IniParser& iniConfig,
                                      const std::string& section)
{
    std::string currentSection = section;

    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    const bool loadInMemory = iniConfig.getProperty
                              <bool>("LoadInMemory", false);

    std::shared_ptr<DIR_Database> database = std::make_shared
        <DIR_Database>(loadInMemory);

    do {
        if (!iniConfig.currentSection(currentSection)) {
            throw std::runtime_error("Missing ["
                                     + currentSection + "] section.");
        }

        const std::string dataPath
            = Utils::expandEnvVars(iniConfig.getProperty<std::string>
                                   ("DataPath"));
        const int depth = iniConfig.getProperty<int>("Depth", 1);
        const std::string labelName = iniConfig.getProperty
                                      <std::string>("LabelName", "");
        const int labelDepth = iniConfig.getProperty<int>("LabelDepth", 1);
        const std::string roiFile = Utils::expandEnvVars(
            iniConfig.getProperty<std::string>("ROIFile", ""));
        const bool perLabel = iniConfig.getProperty
                              <bool>("PerLabelPartitioning", true);
        const double learn = iniConfig.getProperty<double>("Learn");
        const double validation = iniConfig.getProperty<double>("Validation",
                                                                0.0);
        const double test = (perLabel)
            ? iniConfig.getProperty<double>("Test", 1.0 - learn - validation)
            : iniConfig.getProperty<double>("Test", 0.0);

        if (iniConfig.isProperty("ValidExtensions")) {
            database->setValidExtensions(
                iniConfig.getProperty<std::vector<std::string> >
                    ("ValidExtensions", std::vector<std::string>()));
        }

        const std::string loadMore
            = iniConfig.getProperty<std::string>("LoadMore", "");

        database->setParameters(iniConfig.getSection(currentSection, true));
        database->loadDir(dataPath, depth, labelName, labelDepth);

        if (!roiFile.empty())
            database->loadROIs(roiFile, "", true);

        if (perLabel) {
            database->partitionStimuliPerLabel(learn, validation, test, true);
            database->partitionStimuli(0.0, 0.0, 1.0);
        } else {
            database->partitionStimuli(learn, Database::Learn);
            database->partitionStimuli(validation, Database::Validation);
            database->partitionStimuli(test, Database::Test);
        }

        currentSection = loadMore;
    }
    while (!currentSection.empty());

    return database;
}
