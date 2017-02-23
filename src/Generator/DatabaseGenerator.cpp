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

#include "Generator/DatabaseGenerator.hpp"

std::shared_ptr<N2D2::Database>
N2D2::DatabaseGenerator::generate(IniParser& iniConfig,
                                  const std::string& section)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    std::shared_ptr<Database> database;

    if (iniConfig.isProperty("Type")) {
        const std::string type = iniConfig.getProperty<std::string>("Type");
        database = Registrar<DatabaseGenerator>::create(type)(iniConfig,
                                                              section);
    }
    else {
        const std::string labelPath = Utils::expandEnvVars(
            iniConfig.getProperty<std::string>("LabelPath", ""));

        database = std::make_shared<Database>();
        database->load("", labelPath);
    }

    if (iniConfig.isSection(section + ".filterROIs")) {
        iniConfig.currentSection(section + ".filterROIs");

        const std::vector<std::string> labels
            = iniConfig.getProperty<std::vector<std::string> >("Labels");
        const bool filterKeep = iniConfig.getProperty<bool>("FilterKeep", true);
        const bool removeStimuli = iniConfig.getProperty
                                   <bool>("RemoveStimuli", true);

        database->filterROIs(labels, filterKeep, removeStimuli);
    }

    if (iniConfig.isSection(section + ".slicing")) {
        iniConfig.currentSection(section + ".slicing");

        const unsigned int width = iniConfig.getProperty<unsigned int>("Width");
        const unsigned int height = iniConfig.getProperty
                                    <unsigned int>("Height");
        const unsigned int strideX = iniConfig.getProperty
                                     <unsigned int>("StrideX");
        const unsigned int strideY = iniConfig.getProperty
                                     <unsigned int>("StrideY");
        const Database::StimuliSetMask applyTo
            = iniConfig.getProperty<Database::StimuliSetMask>("ApplyTo");

        database->extractSlices(width, height, strideX, strideY, applyTo);
    }

    return database;
}
