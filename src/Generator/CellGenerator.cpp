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

#include "Generator/CellGenerator.hpp"

std::string N2D2::CellGenerator::mDefaultModel = "Frame";

std::shared_ptr<N2D2::Cell>
N2D2::CellGenerator::generate(Network& network,
                              StimuliProvider& sp,
                              const std::vector
                              <std::shared_ptr<Cell> >& parents,
                              IniParser& iniConfig,
                              const std::string& section)
{
    if (!iniConfig.currentSection(section, false))
        throw std::runtime_error("Missing [" + section + "] section.");

    const std::string type = iniConfig.getProperty<std::string>("Type");
    return Registrar
        <CellGenerator>::create(type)(network, sp, parents, iniConfig, section);
}

void N2D2::CellGenerator::postGenerate(const std::shared_ptr<Cell>& cell,
                                       const std::shared_ptr<DeepNet>& deepNet,
                                       IniParser& iniConfig,
                                       const std::string& section)
{
    if (RegistrarCustom
        <CellGenerator, RegistryPostCreate_T>::exists(cell->getType()
                                                      + std::string("+")))
    {
        RegistrarCustom
            <CellGenerator, RegistryPostCreate_T>::create(cell->getType()
                                                          + std::string("+"))
                (cell, deepNet, iniConfig, section);
    }
}

std::map<std::string, std::string>
N2D2::CellGenerator::getConfig(const std::string& model, IniParser& iniConfig)
{
    const std::string configSection
        = (iniConfig.isProperty("ConfigSection(" + model + ")"))
              ? "ConfigSection(" + model + ")"
              : "ConfigSection";

    std::vector<std::string> configSections;

    if (iniConfig.isProperty(configSection))
        configSections = Utils::split(
            iniConfig.getProperty<std::string>(configSection, ""), ",");

    iniConfig.ignoreProperty("ConfigSection");
    iniConfig.ignoreProperty("ConfigSection(*)");

    std::map<std::string, std::string> params;

    for (std::vector<std::string>::const_iterator itConfig
         = configSections.begin(),
         itConfigEnd = configSections.end();
         itConfig != itConfigEnd;
         ++itConfig) {
        const std::map<std::string, std::string> sectionParams
            = iniConfig.getSection(*itConfig);

        // Use of operator [] to replace existing values.
        // Function insert() never replaces existing values!
        for (std::map<std::string, std::string>::const_iterator it
             = sectionParams.begin(),
             itEnd = sectionParams.end();
             it != itEnd;
             ++it) {
            params[(*it).first] = (*it).second;
        }
    }

    return params;
}
