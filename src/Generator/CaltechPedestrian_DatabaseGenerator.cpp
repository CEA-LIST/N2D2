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

#include "Generator/CaltechPedestrian_DatabaseGenerator.hpp"

N2D2::Registrar<N2D2::DatabaseGenerator>
N2D2::CaltechPedestrian_DatabaseGenerator::mRegistrar(
    "CaltechPedestrian_Database",
    N2D2::CaltechPedestrian_DatabaseGenerator::generate);

std::shared_ptr<N2D2::CaltechPedestrian_Database>
N2D2::CaltechPedestrian_DatabaseGenerator::generate(IniParser& iniConfig,
                                                    const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const double validation = iniConfig.getProperty<double>("Validation", 0.0);
    const bool singleLabel = iniConfig.getProperty<bool>("SingleLabel", true);
    const bool incAmbiguous = iniConfig.getProperty
                              <bool>("IncAmbiguous", false);

    const std::string dataPath
        = Utils::expandEnvVars(iniConfig.getProperty<std::string>(
            "DataPath", N2D2_DATA("CaltechPedestrians/data-USA/images")));
    const std::string labelPath
        = Utils::expandEnvVars(iniConfig.getProperty<std::string>(
            "LabelPath", N2D2_DATA("CaltechPedestrians/data-USA/annotations")));

    std::shared_ptr<CaltechPedestrian_Database> database = std::make_shared
        <CaltechPedestrian_Database>(validation, singleLabel, incAmbiguous);
    database->setParameters(iniConfig.getSection(section, true));
    database->load(dataPath, labelPath);
    return database;
}
