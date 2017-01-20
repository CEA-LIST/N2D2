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

#include "Generator/EnvironmentGenerator.hpp"

std::shared_ptr<N2D2::Environment>
N2D2::EnvironmentGenerator::generate(Network& network,
                                     Database& database,
                                     IniParser& iniConfig,
                                     const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const unsigned int sizeX = iniConfig.getProperty<unsigned int>("SizeX");
    const unsigned int sizeY = iniConfig.getProperty<unsigned int>("SizeY");
    const unsigned int nbChannels = iniConfig.getProperty
                                    <unsigned int>("NbChannels", 1U);
    const unsigned int batchSize = iniConfig.getProperty
                                   <unsigned int>("BatchSize", 1U);
    const bool compositeStimuli = iniConfig.getProperty
                                  <bool>("CompositeStimuli", false);
    const std::string cachePath = iniConfig.getProperty
                                  <std::string>("CachePath", "");

    std::shared_ptr<Environment> env(new Environment(network,
                                                     database,
                                                     sizeX,
                                                     sizeY,
                                                     nbChannels,
                                                     batchSize,
                                                     compositeStimuli));
    env->setCachePath(cachePath);

    iniConfig.setProperty("_EpochSize", database.getNbStimuli(Database::Learn));

    const std::string configSection = iniConfig.getProperty
                                      <std::string>("ConfigSection", "");

    if (!configSection.empty())
        env->setParameters(iniConfig.getSection(configSection));

    generateSubSections(env, iniConfig, section);
    return env;
}
