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
        const std::string roiDir = Utils::expandEnvVars(
            iniConfig.getProperty<std::string>("ROIDir", ""));
        std::vector<std::string> roiExt = iniConfig.getProperty
            <std::vector<std::string> >("ROIExtension",
                                        std::vector<std::string>(1, "json"));
        std::transform(roiExt.begin(),
                       roiExt.end(),
                       roiExt.begin(),
                       Utils::expandEnvVars);

        const bool perLabel = iniConfig.getProperty
                              <bool>("PerLabelPartitioning", true);
        const bool equivLabel = iniConfig.getProperty
                              <bool>("EquivLabelPartitioning", true);
        const double learn = iniConfig.getProperty<double>("Learn");
        const double validation = iniConfig.getProperty<double>("Validation",
                                                                0.0);

        double test = 0.0;

        if (iniConfig.isProperty("Test"))
            test = iniConfig.getProperty<double>("Test");

        if (iniConfig.isProperty("IgnoreMasks")) {
            database->setIgnoreMasks(
                iniConfig.getProperty<std::vector<std::string> >
                    ("IgnoreMasks", std::vector<std::string>()));
        }

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

        if (!roiDir.empty())
            database->loadROIsDir(roiDir, roiExt, depth);

        if (perLabel) {
            if (learn + validation > 1.0) {
                std::stringstream errorMsg;
                errorMsg << "DIR_DatabaseGenerator: Learn (" << learn << ") + "
                    "Validation (" << validation << ") cannot be > 1.0";

                throw std::runtime_error(errorMsg.str());
            }

            if (!iniConfig.isProperty("Test"))
                test = 1.0 - learn - validation;

            database->partitionStimuliPerLabel(learn, validation, test,
                                               equivLabel);
            database->partitionStimuli(0.0, 0.0, 1.0);
        } else {
            if (!iniConfig.isProperty("Test"))
                test = database->getNbStimuli() - learn - validation;

            database->partitionStimuli(learn, Database::Learn);
            database->partitionStimuli(validation, Database::Validation);
            database->partitionStimuli(test, Database::Test);
        }

        currentSection = loadMore;
    }
    while (!currentSection.empty());

    database->logPartition("db_partition");

    return database;
}
